#!/bin/bash
# Build the presentation versions of the report's Table 1, Figure 1 and
# Figure 2: 16:9 proportions, vplot colours, legends below the panels, no
# grids.  These are separate objects from the report's own figures, which are
# built by the pipeline and are not touched here.
#
# Requires the `vplot` package (pip install "vplot>=1.0").
set -e
cd "$(dirname "$0")"
mkdir -p ../Plot/slides

python3 plotSlideRadiationSchemes.py \
  --radiation-json ../compareRadiationSchemes/radiationSchemes.json \
  --co2-json ../explorations/olrCO2Dependence.json \
  --out ../Plot/slides/slideRadiationSchemes.pdf

python3 plotSlideAlbedoSchemes.py \
  --albedo-json ../compareAlbedoSchemes/albedoSchemes.json \
  --out ../Plot/slides/slideAlbedoSchemes.pdf

pdflatex -interaction=nonstopmode -halt-on-error slideStructureTable.tex >/dev/null
pdftoppm -r 220 -png -singlefile slideStructureTable.pdf ../Plot/slides/slideStructureTable
cp slideStructureTable.pdf ../Plot/slides/slideStructureTable.pdf
rm -f slideStructureTable.aux slideStructureTable.log slideStructureTable.out slideStructureTable.pdf

echo "wrote ../Plot/slides/:"
ls -1 ../Plot/slides/
