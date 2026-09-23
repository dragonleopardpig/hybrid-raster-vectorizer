# Hybrid Raster Vectorizer

Reconstructs a scanned scientific figure as clean, editable SVG — semantic
axes, ticks, fitted curves and typeset labels — rather than tracing every noisy
pixel boundary.

```sh
./bin/hybrid-vectorizer convert raster.png -o result.svg
```

That works from an ordinary terminal: the wrapper re-enters the dev shell itself
if it is not already in one. Everything runs offline once the OCR model is
present.

## What the converter does

`convert` is a single automatic pass. Each stage hands measurements to the next,
and every decision it makes is recorded in a JSON report beside the SVG.

1. **Preprocess** — greyscale, Otsu binarisation, despeckle, deskew from the
   longest near-horizontal runs. The pen width is measured on the medial axis,
   and almost every later threshold is expressed as a multiple of it.
2. **Rules** — long straight runs are found by morphological opening, giving
   each axis its position, extent and thickness. Walking inward from each tip
   while the perpendicular thickness grows detects arrowheads and measures them.
3. **Ticks** — marks adjacent to a rule and no wider than the pen. A crossing
   axis and an arrowhead both flare like a tick, so both are excluded by
   position. Surviving marks are fitted to a lattice and snapped to it when the
   residual is small.
4. **Curves** — axes, arrowheads and ticks are erased, and what remains is
   traced along its centreline, per column where the stroke is a function of x
   and along the medial axis otherwise. Strokes that an erased axis cut apart
   are rejoined.
5. **Fitting** — the trace is matched against straight lines, polynomials and
   sinusoids (period by spectrum, then a bracketed minimisation), and fitted
   with cubic Béziers by Schneider's algorithm to a tolerance set as a fraction
   of the pen width. The analytic reading is recorded on the path either way;
   `--idealise` redraws from it instead of from the ink.
6. **Text** — leftover ink is grouped into labels. Fraction bars are found
   structurally, by being the only rule with ink both above and below, which is
   what separates them from an equals sign or a leading minus. Each bar claims
   its own numerator and denominator, so adjacent tick labels cannot run
   together.
7. **Reading** — labels are routed to Tesseract or, through a persistent worker,
   to PP-FormulaNet. A confident prose reading wins; anything else is treated as
   mathematics.
8. **Typesetting** — the LaTeX is parsed and laid out using the real advance
   widths of the matched font, then written as positioned SVG text. Repeated
   structures on one row are set in a single size.
9. **Verification** — the finished SVG is rasterised with resvg and compared
   against the original ink, and the agreement is reported.

## What it gets right, and what it does not

On `examples/interference/raster.png`, a scan of a mid-century textbook figure:

| | result |
|---|---|
| ink agreement | recall 0.979, precision 0.959 within 3px |
| axes | both found, with the arrowhead on the correct end of each |
| ticks | all 10, spacing 96.93px, snapped to a lattice |
| curve | one stroke, 47 Bézier segments, recognised as a sinusoid |
| labels | 11 blocks, all six tick fractions bound correctly |

The curve's analytic residual is 10.8px against an amplitude of 264px. That is
not a fitting failure: the figure is hand-drawn, and its humps drift by about
±10px from a true cos². A single hump fits to 3.6px, and the measured half-width
of 95.0px matches the 97.0px a cos² of the detected period predicts. The
reconstruction therefore follows the ink, and reports the analytic reading
separately.

**Text is where the limits are.** The recogniser reads the δ in this figure as
`s` and one of them as `B`, and the converter does not silently fix either.

Correcting a glyph by re-rendering candidates in installed fonts and comparing
shapes was implemented and then **measured against known-correct glyphs: it
chose the right letter in about one case in six.** The scanned typeface is not
installed, and the margin between a candidate and its confusable twin is smaller
than the difference between two typefaces, so the comparison ranks fonts rather
than letterforms. It is available as `--substitute-glyphs` and is off by
default, because a confident wrong answer is worse than an honest uncertain one.

What *is* used instead:

- **Structure from the ink.** The number of glyphs genuinely drawn off the
  baseline is measured; when a reading claims more subscripts than that, the
  surplus is demoted. This is what corrects `a_{\pi}` to `aπ`.
- **Agreement between repeated symbols.** Comparing one scanned glyph with
  another from the *same* figure has no typeface mismatch, so identical symbols
  are clustered (complete-link, so every member resembles every other) and made
  to read alike. A rewrite needs at least three agreeing symbols.
- **Honest flagging.** Glyphs in a confusable set are listed in the report, as
  are glyphs that touch a neighbour and so could not be checked against the ink
  at all. `--raster-fallback` embeds the original pixels for any label below the
  confidence threshold.

Font matching ranks installed families against confidently-read prose. On the
example it picks TeX Gyre Bonum at a score of 0.53 — a weak match, reported as
such. Use `--font` to override it.

## Reproduce

```sh
devenv shell          # or: nix develop
make convert
make test
```

`make convert` writes `build/auto.svg`, `build/auto.report.json` and
`build/auto.png`. The OCR model must already be present:

```sh
formulaocr-offline --download-model
```

## The earlier hand-written path

`render` still builds an SVG from a hand-written JSON specification, and
`inspect` still runs OCR over crops named in one. `examples/interference/spec.json`
is the manually corrected reconstruction of the same figure, kept as a reference
for what a fully correct result looks like.

```sh
hybrid-vectorizer render examples/interference/spec.json -o build/result.svg
```

## Options

| flag | effect |
|---|---|
| `--idealise` | redraw curves from the fitted analytic model, not the traced ink |
| `--font FAMILY` | force a font family instead of matching one |
| `--bezier-tolerance F` | curve fit tolerance as a fraction of the pen width (default 0.25) |
| `--confidence F` | below this, a label is listed for review (default 0.55) |
| `--raster-fallback` | embed original pixels for labels below that threshold |
| `--substitute-glyphs` | enable font-template glyph correction (measured unreliable) |
| `--no-deskew`, `--no-formula-ocr`, `--no-verify` | skip a stage |

## Still missing

- Filled regions, hatching, legends, data markers and embedded images.
- Multi-line prose blocks and rotated text.
- Occlusion recovery beyond rejoining strokes an axis cut apart.
- Curve families other than lines, polynomials and sinusoids.
- Glyph identification good enough to correct a recogniser, which needs the
  scanned typeface itself rather than whatever fonts happen to be installed.
