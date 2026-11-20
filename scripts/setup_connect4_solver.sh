#!/usr/bin/env bash
# Fetch, build and install Pascal Pons' Connect Four solver (AGPL-3.0) into
# third_party/connect4. It is kept out of this repo and called as a separate
# process by cs440_final_project.connect4.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=third_party/connect4
if [ ! -d "$DIR" ]; then
  git clone --depth 1 https://github.com/PascalPons/connect4 "$DIR"
fi
make -C "$DIR" -s c4solver
if [ ! -f "$DIR/7x6.book" ]; then
  curl -fL -o "$DIR/7x6.book" https://github.com/PascalPons/connect4/releases/download/book/7x6.book
fi
echo "4" | "$DIR/c4solver" -a -b "$DIR/7x6.book" >/dev/null 2>&1 && echo "solver OK: $DIR/c4solver"
