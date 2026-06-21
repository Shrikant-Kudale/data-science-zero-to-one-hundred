# Day 21 — Publication-Ready Data Visualization

## Principles (Tufte / Cleveland–McGill)
- Perceptual accuracy ranking: position on common scale > length > angle/slope > area > color saturation. Prefer bars/dots/lines over pies and 3-D.
- **Data-ink ratio:** remove chart junk (heavy grids, borders, redundant legends).
- Title states the *takeaway* ("Retention drops 40% after month 2"), axes labeled with units, direct labels over legends.
- Bars must start at zero; lines need not. Don't use dual axes casually. Use log scale for multiplicative data.
- Color: sequential (ordered), diverging (centered at meaningful midpoint), qualitative (≤8 categories). Colorblind-safe palettes (viridis, cividis, ColorBrewer); never encode meaning by red/green alone.
- Show uncertainty (CIs, bands), sample size, and distributions rather than only means (violin/box/strip; "dynamite plots" hide data).

## Choosing a chart
| Question | Chart |
|---|---|
| Distribution | histogram/KDE/ECDF/box/violin |
| Comparison | bar/dot plot (sorted) |
| Trend | line (+ CI band) |
| Relationship | scatter (+ regression, hexbin for big data) |
| Composition | stacked bars (few parts), treemap |
| Correlation matrix | heatmap (diverging, clustered) |
| Many groups | small multiples (facets) with shared axes |
Overplotting: alpha, hexbin, 2D KDE, sampling.

## Tools
Matplotlib (full control, object-oriented `fig, ax`), Seaborn (statistical defaults, tidy data), Plotly (interactive HTML, dashboards), Altair (grammar of graphics). Export vector (SVG/PDF) for papers, 300-dpi PNG for slides; set fonts and figure size to final print size.

## FAANG Q&A
**Q1. Show a metric over time with a spike to execs?** Line with annotation of the event, baseline band, takeaway title.
**Q2. Why avoid pie charts?** Angles/areas are hard to compare; sorted bar is better.
**Q3. 1M points scatter?** Hexbin/2D histogram/datashader; alpha; sample.
**Q4. How to chart uncertainty in A/B results?** Forest plot of lift with CIs and a zero line.
**Q5. Dashboard vs static chart?** Dashboards for monitoring repeated questions; static narrative for decisions.
