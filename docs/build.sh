#!/usr/bin/env bash
# Build the passess documentation site.
#
# 1. Convert jupytext examples (py:percent) to .ipynb under docs/examples/
#    and execute them so figures/outputs are embedded in the notebook.
# 2. Generate API reference pages with quartodoc.
# 3. Render the Quarto site.
#
# Usage: docs/build.sh [render|preview]
# Defaults to `render`. Pass `preview` to open a live-reloading preview.

set -euo pipefail

here="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
repo_root="$(cd -- "$here/.." >/dev/null 2>&1 && pwd)"
mode="${1:-render}"

mkdir -p "$here/examples"

for src in "$repo_root/examples/"*.py; do
    name="$(basename "$src" .py)"
    out="$here/examples/$name.ipynb"
    echo "jupytext $src -> $out"
    jupytext --to ipynb --execute --output "$out" "$src"
done

echo "quartodoc build"
(cd "$here" && quartodoc build)

echo "quarto $mode"
(cd "$here" && quarto "$mode")
