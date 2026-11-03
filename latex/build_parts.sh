#!/usr/bin/env bash
# Build the full notes (notes.tex -> notes.pdf, notes.sec, notes.aux), then the parts of the notes
# used by each code path (parts/<case>.tex -> parts/pdf/<case>.pdf). A part keeps the numbering
# of the full notes and needs notes.sec and notes.aux, so the full notes are always built first.
# Usage (from anywhere):  latex/build_parts.sh [case ...]    default: every parts/*.tex
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error notes.tex
if [ $# -eq 0 ]; then set -- $(ls parts/*.tex | xargs -n1 basename | sed 's/\.tex$//'); fi
for c in "$@"; do
    [ -f "parts/$c.tex" ] || { echo "no parts/$c.tex" >&2; exit 1; }
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=parts/pdf "parts/$c.tex"
done
