#!/bin/bash
# Compile the three-code comparison report to PDF and place it in Plot/.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error filletThreeCodeComparison.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error filletThreeCodeComparison.tex >/dev/null
mkdir -p ../Plot
cp filletThreeCodeComparison.pdf ../Plot/filletThreeCodeComparison.pdf
echo "wrote ../Plot/filletThreeCodeComparison.pdf"
