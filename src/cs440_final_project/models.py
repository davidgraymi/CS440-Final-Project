"""Recursive thinking model (TRM) and comparison baselines.

The TRM here matches the official implementation
(github.com/SamsungSAILMontreal/TinyRecursiveModels) closely enough to load its
checkpoints (see `load_official_trm`): parameter-free post-norm RMSNorm, SwiGLU
widths rounded up to multiples of 256, a 16-token learned prefix (the
"puzzle embedding") whose first position feeds the halting head, and the
rotate-half RoPE convention.

All models share one interface so the training loop treats them identically:

    carry = model.init_carry(batch_size, device)
    for _ in range(model.n_sup):                      # supervision steps
        carry, logits, q_halt = model.step(inputs, carry)

* `TRM`    - Tiny Recursive Model (Jolicoeur-Martineau, 2025). One small
             network, reused: a latent state z is refined n times from
             (x, y, z), then the answer y is refined from (y, z). That cycle
             repeats T times (only the last with gradients) per supervision
             step, and the detached (y, z) carry into the next step.
             With `n=0` it becomes the "deep supervision only" ablation:
             no latent z, just y <- net(y + x) once per recursion.
* `Direct` - Conventional one-pass transformer (no recursion, one step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

NORM_EPS = 1e-5


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 512
    n_layers: int = 2
    n_heads: int = 8
    mixer: str = "attn"  # "attn" (self-attention + RoPE) or "mlp" (MLP over positions)
    expansion: float = 4.0
    n_prefix: int = 16  # learned prefix tokens; position 0 feeds the halting head
    # TRM recursion
    n: int = 6  # latent (z) updates per recursion   (official: L_cycles)
    T: int = 3  # recursions per supervision step     (official: H_cycles)
    n_sup: int = 16  # max supervision steps           (official: halt_max_steps)


# ----------------------------------------------------------------- components


def rms_norm(x):
    """Parameter-free RMSNorm over the last axis, computed in float32."""
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)).to(dtype)


def _round_up(a: int, b: int) -> int:
    return -(a // -b) * b


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float):
        super().__init__()
        hidden = _round_up(round(expansion * dim * 2 / 3), 256)
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Bidirectional multi-head self-attention with rotary positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        inv = 1.0 / 10000.0 ** (torch.arange(0, self.dh, 2).float() / self.dh)
        ang = torch.outer(torch.arange(cfg.seq_len + cfg.n_prefix).float(), inv)
        ang = torch.cat((ang, ang), dim=-1)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        cos, sin = self.cos.to(q.dtype), self.sin.to(q.dtype)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """Post-norm block: mix tokens, then mix channels.

    MLP mixer (TRM-MLP): a SwiGLU across the sequence axis. As in the official
    code, its residual + norm happen in the transposed (B, D, L) layout, i.e.
    that RMSNorm normalizes over positions, not channels."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mixer = cfg.mixer
        if cfg.mixer == "attn":
            self.attn = Attention(cfg)
        else:
            self.mlp_t = SwiGLU(cfg.seq_len + cfg.n_prefix, cfg.expansion)
        self.mlp = SwiGLU(cfg.d_model, cfg.expansion)

    def forward(self, h):
        if self.mixer == "attn":
            h = rms_norm(h + self.attn(h))
        else:
            t = h.transpose(1, 2)
            h = rms_norm(t + self.mlp_t(t)).transpose(1, 2)
        return rms_norm(h + self.mlp(h))


class Net(nn.Module):
    """A stack of blocks applied to `state + injection`."""

    def __init__(self, cfg: ModelConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(Block(cfg) for _ in range(n_layers))

    def forward(self, h, inject=None):
        if inject is not None:
            h = h + inject
        for layer in self.layers:
            h = layer(h)
        return h


class _Base(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.scale = math.sqrt(cfg.d_model)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.trunc_normal_(self.embed.weight, std=1.0 / self.scale)
        # Learned prefix vector (official: zero-init "puzzle embedding" for the
        # single puzzle id). Occupies prefix position 0; the rest are zeros.
        self.prefix = nn.Parameter(torch.zeros(cfg.d_model))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed_inputs(self, tokens):
        x = self.embed(tokens)
        if self.cfg.n_prefix:
            pre = x.new_zeros(x.shape[0], self.cfg.n_prefix, self.cfg.d_model)
            pre[:, 0] = self.prefix
            x = torch.cat((pre, x), dim=1)
        return x * self.scale

    def answer_logits(self, h):
        return self.lm_head(h[:, self.cfg.n_prefix:])


# ----------------------------------------------------------------------- TRM


class TRM(_Base):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.net = Net(cfg, cfg.n_layers)
        self.n_sup = cfg.n_sup
        # Fixed (non-learned) initial states, as in TRM.
        self.register_buffer("y_init", nn.init.trunc_normal_(torch.empty(cfg.d_model), std=1.0))
        self.register_buffer("z_init", nn.init.trunc_normal_(torch.empty(cfg.d_model), std=1.0))
        # Halting head: logit that "the current answer is fully correct".
        self.q_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)  # start out never halting

    def init_carry(self, batch_size: int, device):
        shape = (batch_size, self.cfg.seq_len + self.cfg.n_prefix, self.cfg.d_model)
        return (self.y_init.to(device).expand(shape).clone(),
                self.z_init.to(device).expand(shape).clone())

    def latent_recursion(self, x, y, z):
        if self.cfg.n == 0:  # ablation: no latent reasoning state
            return self.net(y, x), z
        for _ in range(self.cfg.n):
            z = self.net(z, x + y)  # think: refine latent from question + answer
        y = self.net(y, z)  # act: refine answer from latent
        return y, z

    def step(self, tokens, carry):
        x = self.embed_inputs(tokens)
        y, z = carry
        with torch.no_grad():
            for _ in range(self.cfg.T - 1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        q_halt = self.q_head(y[:, 0]).float().squeeze(-1)
        return (y.detach(), z.detach()), self.answer_logits(y), q_halt

    def effective_depth(self) -> int:
        """Layer applications per supervision step."""
        per_rec = self.cfg.n_layers * (self.cfg.n + 1 if self.cfg.n else 1)
        return per_rec * self.cfg.T


# -------------------------------------------------------------------- Direct


class Direct(_Base):
    """Standard transformer: one forward pass, no recursion, no halting."""

    n_sup = 1

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.net = Net(cfg, cfg.n_layers)

    def init_carry(self, batch_size: int, device):
        return None

    def step(self, tokens, carry):
        return None, self.answer_logits(self.net(self.embed_inputs(tokens))), None

    def effective_depth(self) -> int:
        return self.cfg.n_layers


# -------------------------------------------------------------------- losses


def stablemax_cross_entropy(logits, labels):
    """Cross-entropy under stablemax, s(x) = x+1 (x>=0) or 1/(1-x) (x<0).
    HRM/TRM use it for stability on small-data puzzle training.
    (Computed in float32: MPS has no float64.)"""
    logits = logits.float()
    s = torch.where(logits >= 0, logits + 1, 1 / (1 - logits))
    logp = torch.log(s / s.sum(-1, keepdim=True))
    return -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def token_loss(logits, labels, kind: str = "stablemax"):
    """Per-example mean token loss, shape (B,)."""
    if kind == "stablemax":
        return stablemax_cross_entropy(logits, labels).mean(-1)
    return F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none").mean(-1)


def build_model(kind: str, cfg: ModelConfig) -> nn.Module:
    return {"trm": TRM, "direct": Direct}[kind](cfg)


# ------------------------------------------------- official checkpoint import


def convert_official_state_dict(sd: dict) -> dict:
    """Map an official TRM state dict (as saved by its training code) to ours."""
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("_orig_mod.").removeprefix("model.").removeprefix("inner.")
        k = (k.replace("L_level.", "net.")
              .replace("embed_tokens.embedding_weight", "embed.weight")
              .replace("H_init", "y_init").replace("L_init", "z_init")
              .replace("gate_up_proj", "gate_up").replace("down_proj", "down")
              .replace("self_attn.qkv_proj", "attn.qkv").replace("self_attn.o_proj", "attn.out"))
        v = v.float()
        if k == "puzzle_emb.weights":
            k, v = "prefix", v[0]
        elif k.startswith("q_head."):  # official head has [halt, continue]; keep halt
            v = v[:1]
        out[k] = v
    return out


def load_official_trm(ckpt_path: str, cfg: ModelConfig) -> TRM:
    model = TRM(cfg)
    sd = convert_official_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.load_state_dict(sd, strict=True)
    return model
