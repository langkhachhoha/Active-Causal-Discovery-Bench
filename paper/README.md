# RauMa — paper source

`rauma_neurips2026.tex` is the Study 1 write-up (NeurIPS 2026 template, anonymous
submission option). Everything it quotes comes from the committed run logs in `study1/`,
not from a notebook that no longer exists.

```bash
# build (needs the figures, which live one level up)
pdflatex rauma_neurips2026.tex && pdflatex rauma_neurips2026.tex
# or, with no local TeX installation:
tectonic -X compile rauma_neurips2026.tex --outdir .

# re-derive every number in the paper from the raw logs
python paper/verify_numbers.py        # run from the repo root
```

The bibliography is inline (`thebibliography`), so no BibTeX pass is needed. Figures are
pulled from `../figures/*.pdf` via `\graphicspath`, so the file also builds if you copy it
to the repo root. Figure 1 is drawn in TikZ inside the document; there is no external
method-figure asset.

## Regenerating the figures

```bash
python scripts/make_rauma_figures.py    --result-dir result/study1 --out-dir figures  # F2, scaling, appendix
python scripts/rauma_edge_audit.py      --study-dir study1 --out figures/edge_audit.json
python scripts/make_rauma_figures_v2.py --study-dir study1 --out-dir figures          # F3, F4, F5
```

`rauma_edge_audit.py` replays the deterministic parts of each instance (the true DAG and
the PC front-end, both fixed by the seed) and classifies every arrow the readouts
submitted, reading the LLM-produced graphs back out of `events.jsonl`. It needs the
project environment (`causallearn`), unlike the plotting scripts.

## What the paper is allowed to claim

`verify_numbers.py` prints each table and each paired test side by side with the value in
the text. Four constraints the write-up respects deliberately:

- **0.836 is not a ceiling.** `oracle+meek` scores 0.836 because truth-aware selection
  minimises the *number* of experiments (1.60 vs 1.78), so it collects less evidence;
  `llm+meek` reaches 0.861. The paper calls it the truth-aware *reference*, and the
  readout gap is stated as measured against a beatable opponent.
- **Selection is not free.** In the main run the selector contrasts are inside the noise,
  but under a tight budget and at n_obs ∈ {60, 1000} every informed selector beats random
  at p < 0.05 (+0.03 to +0.08 F1). The paper says selection is worth an order of magnitude
  less than interpretation, not that it is worth nothing.
- **The ablation instances are a separate draw.** `ablation_tightbudget` shares the main
  seed map and pairs with it. `ablation_n60`, `ablation_n1000` and `ablation_rawevidence`
  share a *different* seed map with each other, so contrasts inside those runs are paired
  and comparisons against the main run are unpaired (Mann–Whitney) and labelled as such.
- **The token ratio measures the scaffold, not the model.** The unscaffolded agent is
  served raw rows while the scaffolded arms get sufficient statistics. Section 5.3 isolates
  that factor on its own (`ablation_rawevidence`) and finds it worth nothing.

## Runs behind the paper

| directory | episodes | what changes |
|---|---:|---|
| `study1/main` | 720 | the 5×2 grid plus the unscaffolded agent, 40 instances |
| `study1/ablation_tightbudget` | 240 | budget cut from \|I*\|+1 to \|I*\| |
| `study1/ablation_n60` | 360 | n_obs/n_int = 60/40, levels 1–2 |
| `study1/ablation_n1000` | 360 | n_obs/n_int = 1000/500, levels 1–2 |
| `study1/ablation_rawevidence` | 120 | LLM readout served raw rows instead of summaries |

1,800 episodes, 12.7M tokens, \$2.39, zero failed LLM calls.
