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
4. **Areas** — filled and ruled regions are claimed before anything else,
   because a long run through a filled shape is indistinguishable from an axis
   and the parallel strokes of ruling are indistinguishable from a dozen curves.
   Solidity is measured by distance from the background rather than by filling
   the outline, which for anything thin returns the shape again. Ruling reports
   its angle and spacing and becomes an SVG `<pattern>`; a drawn frame around it
   is kept, and one that is only where the ruling stops is not invented.
5. **Marker series** — congruent marks that stand alone are grouped into a
   series and named (circle, rectangle, triangle), then emitted once as a
   `<symbol>` and placed with `<use>`. Once the shape is known from the copies
   out in the plot, the legend's sample is claimed by resemblance, since that one
   never stands alone.
6. **Curves** — axes, arrowheads and ticks are erased, and what remains is
   traced along its centreline, per column where the stroke is a function of x
   and along the medial axis otherwise. Strokes that an erased axis cut apart
   are rejoined.
7. **Fitting** — the trace is matched against straight lines, polynomials and
   sinusoids (period by spectrum, then a bracketed minimisation), and fitted
   with cubic Béziers by Schneider's algorithm to a tolerance set as a fraction
   of the pen width. The analytic reading is recorded on the path either way;
   `--idealise` redraws from it instead of from the ink.
8. **Text** — leftover ink is grouped into labels. Fraction bars are found
   structurally, by being the only rule with ink both above and below, which is
   what separates them from an equals sign or a leading minus. Each bar claims
   its own numerator and denominator, so adjacent tick labels cannot run
   together.
9. **Reading** — labels are routed to Tesseract or, through a persistent worker,
   to PP-FormulaNet. A confident prose reading wins; anything else is treated as
   mathematics.
10. **Typesetting** — the LaTeX is parsed and laid out using the real advance
   widths of the matched font, then written as positioned SVG text. Repeated
   structures on one row are set in a single size.
11. **Verification** — the finished SVG is rasterised with resvg and compared
   against the original ink, and the agreement is reported.

The page is **transparent** and every mark paints with `currentColor`, so an
inline SVG simply takes the colour of the text around it. Viewed on its own the
file defaults to dark ink and switches to light ink under
`prefers-color-scheme: dark`; a label kept as raster pixels is inverted to match.
`--background COLOR` paints a solid page instead.

## What it gets right, and what it does not

`examples/mixed/figure.png` is drawn by `tools/make_mixed_example.py` with known
contents, because the interference scan is all strokes and cannot show whether
areas and markers are found — or whether looking for them misfires.

| | result |
|---|---|
| ink agreement | recall 0.973, precision 0.988 within 3px |
| filled area | found, boundary fitted |
| ruled area | found at 45° and 9.9px, as a `<pattern>`, frame kept |
| marker series | both found, named circle and rectangle, filled and hollow |
| axes | one arrowhead each, on the correct end |

The same detectors find **no** areas and **no** marker series in the
interference figure, which is the property that matters: a test for a feature a
drawing does not have must come back empty. A test pins that.

On `examples/interference/raster.png`, a scan of a mid-century textbook figure:

| | result |
|---|---|
| ink agreement | recall 0.986, precision 0.956 within 3px |
| axes | both found, with the arrowhead on the correct end of each |
| ticks | all 10, spacing 96.93px, snapped to a lattice |
| curve | one stroke, 47 Bézier segments, recognised as a sinusoid |
| labels | 11 blocks, all six tick fractions bound correctly |
| label text | 4 of 11 exactly right, 0.925 of characters right |

The curve's analytic residual is 10.8px against an amplitude of 264px. That is
not a fitting failure: the figure is hand-drawn, and its humps drift by about
±10px from a true cos². A single hump fits to 3.6px, and the measured half-width
of 95.0px matches the 97.0px a cos² of the detected period predicts. The
reconstruction therefore follows the ink, and reports the analytic reading
separately.

**Text is where the limits are.** Every remaining character error in the example
is one symbol: the recogniser reads δ as `s`, once as `π` and once as `B`. The
converter does not silently fix any of them.

Two ways of fixing it were built and measured, and **both are worse than leaving
the reading alone**, so both are off by default.

*Per glyph, against installed fonts* (`--substitute-glyphs`): re-render each
candidate letter and keep the best match. Measured against known-correct glyphs
it chose the right letter in about **one case in six**. The scanned typeface is
not installed, and the gap between a candidate and its confusable twin is
smaller than the gap between two typefaces, so the score ranks fonts rather than
letterforms.

*A second recogniser with Greek* : Tesseract ships 129 languages here,
including `ell` and `grc`. On the isolated glyph it returned `M`, `|`, `X`, `Ι`
and `ν` — **0 of 3** across eight combinations of language, page-segmentation
mode and character whitelist. Its LSTM expects lines of text, not a single
16px mark, so this is not wired in.

*Asking the maths model repeatedly* (`--ensemble N`): read each formula from N
slightly redrawn crops — rescaled, thinned, thickened, rotated, sharpened — and
take the majority. Across 56 readings of this figure **δ never once appeared**.
That is worth knowing: the recogniser is not uncertain about that letterform, it
is confidently wrong, and no amount of resampling will shake it loose. What the
ensemble does give is honest confidence: it marks the equation as read only 60%
of the time and one tick label 80%, which are exactly the two readings that are
genuinely unstable. It costs one OCR pass per variant, so it is off by default.

*The whole alphabet at once* (`--solve-alphabet`): cluster every isolated glyph
in the figure, then assign letters to clusters jointly, one typeface having to
explain all of them. The clustering is sound — it compares ink with ink from the
same scan, so no typeface enters into it — but naming the clusters still needs an
external model. Four scorings were measured against 12 clusters of known
identity:

| scoring | letters right | δ recovered | font chosen |
|---|---|---|---|
| raw overlap, one letter per cluster | 2/12 | 0/3 | Impact |
| raw overlap, independent | 1/12 | 0/3 | Impact |
| z-scored, independent | 1/12 | 0/3 | Noto Sans CJK HK |
| z-scored, one letter per cluster | 3/12 | 0/3 | UbuntuMono Nerd Font |

Trusting the recogniser scores 9/12 on the same clusters. Two premises behind
the joint solve turned out to be false: raw overlap rewards whichever typeface
lays down the most ink, which is why a heavy display face keeps winning; and
"different shapes are different letters" does not hold, because the same letter
lands in several clusters at subscript and full size. This is kept for figures
whose typeface *is* installed, and reported honestly rather than enabled.

What *is* used instead:

- **Order, not position.** Characters are paired with ink in reading order,
  because predicted coordinates drift along a line whenever the typeface in hand
  is not the one that was printed. Where letters ran together, the reading says
  how many are in the mark and the column profile says where to cut — and a cut
  that would pass through a stroke is refused, leaving the glyphs reported as
  unchecked instead of producing pieces that resemble each other whatever they
  came from.
- **Structure from the ink.** A subscript is demoted when *that* glyph's own
  mark is drawn at full size. Counting scripts across a label instead was tried
  and is too blunt: where a subscript touches its base they share one mark, the
  count loses it, and a genuine subscript gets demoted.
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
| `--background COLOR` | paint a solid page instead of leaving it transparent |
| `--ensemble N` | read each formula N ways and report how often they agree |
| `--substitute-glyphs` | per-glyph correction against installed fonts (measured unreliable) |
| `--solve-alphabet` | name every distinct shape at once against installed fonts (measured unreliable) |
| `--no-deskew`, `--no-formula-ocr`, `--no-verify` | skip a stage |

## Still missing

- Legends as a unit. The frame is traced and the entries read, but they are
  not grouped into one object, and a sample mark touching its label becomes one
  component with it, which no amount of proximity reasoning can undo.
- Embedded images, and areas filled with anything other than solid ink or
  evenly spaced ruling.
- Multi-line prose blocks and rotated text.
- Occlusion recovery beyond rejoining strokes an axis cut apart.
- Curve families other than lines, polynomials and sinusoids.
- Glyph identification good enough to correct a recogniser. Four approaches
  have now been measured — per-glyph font templates, a joint alphabet solve, a
  second recogniser with Greek, and an ensemble over redrawn crops — and none
  beats leaving the reading alone. The obstacle is the same each time: no model
  on hand has seen this typeface at this size, and the one that reads the maths
  is confidently wrong rather than uncertain. What would actually help is a
  recogniser that emits per-character alternatives with probabilities, or
  training on the document's own letterforms; guessing harder will not.
- Segmenting letters that genuinely overlap, which a straight vertical cut
  cannot do. The same limit costs the legend sample above.
