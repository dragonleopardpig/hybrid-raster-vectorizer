# Hybrid Raster Vectorizer

This repository is a proof of concept for reconstructing scanned scientific
figures as clean, editable SVG rather than tracing every noisy pixel boundary.

It combines:

1. `formulaocr-offline` (PP-FormulaNet) for mathematical OCR.
2. Tesseract for ordinary captions and labels.
3. ImageMagick for OCR-region extraction.
4. Human or domain-guided correction of ambiguous symbols.
5. Semantic SVG primitives for axes, ticks, arrows and smooth curves.
6. Inkscape for previews and optional conversion of text to portable outlines.

It is intentionally **not** described as a fully automatic general-purpose
vectorizer. The sample OCR read Greek `delta` as `s` and interpreted `a pi` as
a subscript. Repeated tick-label structure and the plotted equation supplied
the information needed to correct those errors. Ordinary outline tracing could
not infer that the axes were straight lines or that the waveform was smooth.

## Example

The example converts `examples/interference/raster.png` into:

- `raster_hybrid.svg`: editable TeX Gyre Termes text and semantic SVG geometry.
- `raster_hybrid_outlined.svg`: the same result with glyphs converted to paths.
- `raster_hybrid_preview.png`: a raster preview for quick inspection.

The curve is one continuous cubic Bezier path. Axes and ticks are SVG `line`
elements, arrowheads are SVG markers, and no bitmap is embedded in either SVG.

## Reproduce

On NixOS:

```sh
nix develop
make inspect
make render
make test
```

Without Nix, install Python 3.11+, ImageMagick, Tesseract, Inkscape and
`formulaocr-offline`, then run the same `make` targets.

The OCR model must already be downloaded:

```sh
formulaocr-offline --download-model
```

## CLI

```sh
PYTHONPATH=src python -m hybrid_vectorizer inspect examples/interference/spec.json

PYTHONPATH=src python -m hybrid_vectorizer render \
  examples/interference/spec.json \
  -o build/result.svg \
  --outlined build/result-outlined.svg \
  --preview build/result.png
```

The JSON specification records OCR crops, corrected labels, fonts and semantic
geometry. New renderers can be added for other figure types without replacing
the deterministic SVG output with a generative approximation.
