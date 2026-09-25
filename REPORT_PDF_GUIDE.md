# Guide: Creating Results-Only PDF Reports

## Purpose

The short PDF reports (`results/REPORT_*_short.pdf`) are **data-only summaries** — they present raw experimental results in a compact, readable format. They do **not** contain analysis, interpretation, or findings. The reader draws their own conclusions from the data.

## What to include

- **Training summary table**: drafter name, cluster/domain info, sample counts, final eval_loss, final top-1 accuracy, training time.
- **Base (untrained) drafter row**: always include the base model (before any fine-tuning) as a row in every evaluation metric matrix. This provides the baseline against which all training improvements are measured. Run eval for the base model on the same clusters/domains as the trained models.
- **Evaluation metric matrices**: one table per metric (overlap area, top-1 match, top-K overlap, KL divergence). Rows = drafters (including base untrained), columns = eval domains/clusters. Bold the best value per column. Use blue shading to highlight high (or low for KL) values.
- **Comparison deltas**: per-cluster drafter vs baseline on its own domain. Show raw values side by side with the delta. No commentary.
- **Training curves**: eval_loss and top-1 accuracy over epochs. Direct labels on the lines, no separate legend box.
- **Footer note**: one line describing the metric definition and eval setup (e.g., "overlap_area = 1 − TVD, teacher-forced on N validation samples").

## What NOT to include

- **No "Key Findings" section.** The report presents data, not conclusions.
- **No analysis or interpretation** of why one model is better than another.
- **No recommendations** for future work.
- **No descriptions of incidents** (OOM kills, restarts, etc.) — these belong in the full markdown report, not the short PDF.

## Style guide

The visual style is defined in `scripts/make_report_pdf.py` (the template). Key design tokens:

| Token | Value | Usage |
|-------|-------|-------|
| `SURFACE` | `#fcfcfb` | Background |
| `INK` | `#0b0b0b` | Primary text, bold values |
| `INK_2` | `#52514e` | Secondary text, normal values |
| `INK_MUTED` | `#8a8983` | Headers, footnotes |
| `RULE` | `#e3e2dd` | Table row separators, grid |
| `BLUE_RAMP` | 5 shades from `#ffffff` to `#9ec5f4` | Cell background shading |

### Tables

- No cell borders — only horizontal rules (thin `RULE` color between rows, solid `INK` at the top).
- First column (drafter names) is bold, left-aligned.
- Data columns are center-aligned.
- Best value per column is **bold**.
- Background shading: map the value range to `BLUE_RAMP` (5 levels). For "lower is better" metrics (KL), invert the mapping.

### Charts

- Clean axis with only left and bottom spines (colored `RULE`).
- Grid lines in `RULE` color, thin.
- Direct labels at the end of each line (no legend box).
- Dash-dot line style for visual distinction between multiple curves.
- Subtitle under each chart title in `INK_MUTED` explaining the metric direction.

### Layout

- A4 page size (`8.27 × 11.69` inches).
- Margins: `[0.07, 0.05, 0.86, 0.90]` (left, bottom, width, height) for the content axes.
- Title block at top: large bold title, subtitle in `INK_2`, horizontal rule.
- One-line config summary under the rule (target model, drafter, loss, etc.) in `INK_MUTED`.
- Footer at very bottom of page 1 with metric definition.

## How to create a new report

1. **Copy the closest existing script** (`make_report_pdf.py` or `generate_wmt16_report_pdf.py`).
2. **Run eval for the base (untrained) drafter** on all eval domains/clusters: `Lite-Mistral-150M-v2-Instruct` is the base model. Eval it using the same pipeline (`src/eval/`) and include the results as a row in every metric matrix.
3. **Replace data only**: update the numpy arrays / table rows with real experimental numbers from training logs and eval CSVs. Do not fabricate or estimate values. Only real measured data goes into the report.
4. **Adjust layout**: if you have more/fewer rows or matrices, split across pages so nothing gets clipped. Test by rendering and visually inspecting.
5. **Run**: `conda run -n domain_sd python scripts/your_report.py`
6. **Verify**: open the PDF to check that all tables render fully, no text is clipped, and numbers match the source data.

## Existing reports

| Script | Output | Content |
|--------|--------|---------|
| `scripts/make_report_pdf.py` | `results/REPORT_domain_drafters_short.pdf` | 3 domain drafters (Understanding, Text Reformulation, Mixed) — 2 pages |
| `scripts/generate_wmt16_report_pdf.py` | `results/REPORT_wmt16_per_cluster_drafters.pdf` | 5 WMT16 per-cluster drafters + TextRef baseline — 3 pages |
