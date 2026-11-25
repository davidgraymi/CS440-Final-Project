"""Tests for the round-robin tournament in c4_tournament.py."""

from cs440_final_project import c4_tournament as T
from cs440_final_project.connect4 import board_from_moves


class FirstLegal:
    """Deterministic stand-in player: the lowest legal column."""

    def moves(self, boards):
        return [b.legal()[0] for b in boards], [1] * len(boards)


class Center:
    def moves(self, boards):
        return [next(c for c in T.CENTER_ORDER if c in b.legal()) for b in boards], [2] * len(boards)


def test_openings_are_distinct_boards():
    # 1 empty board, 7 after one ply, 49 after two; transpositions merge from three plies on
    assert [len(T.openings(n)) for n in range(4)] == [1, 8, 57, 295]
    keys = [board_from_moves(o).key() for o in T.openings(3)]
    assert len(keys) == len(set(keys))


def test_every_game_ends_on_a_win_or_a_full_board():
    opens = T.openings(2)
    games = T.play_pair(FirstLegal(), Center(), ("first", "center"), opens)
    assert len(games) == 2 * len(opens)
    for g in games:
        ms = g["moves"]
        assert ms.startswith(g["opening"])
        before = board_from_moves(ms[:-1])
        last = int(ms[-1]) - 1
        if g["score_a"] == 0.5:
            assert len(ms) == 42
        else:
            assert before.is_winning_move(last)
            # the player who made the last move is the winner
            mover_first = (len(ms) - len(g["opening"])) % 2 == 1
            winner = g["first"] if mover_first else ({"first", "center"} - {g["first"]}).pop()
            assert (g["score_a"] == 1) == (winner == "first")
        # one step count per move each player made
        assert len(g["steps_a"]) + len(g["steps_b"]) == len(ms) - len(g["opening"])
