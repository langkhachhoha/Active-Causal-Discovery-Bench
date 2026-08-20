# Study 2 — PROBE

160 successful episodes from `study2/edits_e2`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec     |
|:-------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:-----------------|:-------------|:-------------|:-------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 0.824 ±0.042  | 0.805 ±0.063   | 0.871 ±0.026  | 2.050 ±0.460 | 0.775 ±0.067 | 2.150 ±0.205         | 979.825 ±46.933  | 98.775 ±5.407       | 1078.600 ±45.639 | 0.000 ±0.000 | 1.000 ±0.000 | 3.283 ±0.497 |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 0.826 ±0.044  | 0.812 ±0.062   | 0.884 ±0.023  | 1.925 ±0.457 | 0.717 ±0.062 | 2.325 ±0.191         | 1461.550 ±63.356 | 294.325 ±36.844     | 1755.875 ±81.052 | 0.001 ±0.000 | 1.000 ±0.000 | 4.109 ±0.484 |
| probe_random_edits | none                         |  40 | 0.793 ±0.051  | 0.798 ±0.069   | 0.864 ±0.026  | 2.200 ±0.450 | 0.712 ±0.060 | 2.325 ±0.215         | —                | —                   | —                | —            | —            | 0.754 ±0.282 |
| probe_skel_only    | none                         |  40 | 0.828 ±0.036  | 0.804 ±0.064   | 0.873 ±0.023  | 1.925 ±0.367 | 0.829 ±0.068 | 1.975 ±0.192         | —                | —                   | —                | —            | —            | 0.605 ±0.287 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source     | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec   | gpt-4o-mini-2024-07-18       |  40 | 0.824 ±0.042  | 0.150 ±0.112          | 0.882 ±0.025            | 48.000 ±0.000  | -0.675 ±0.247      |
| llm_repair + pc_mec   | qwen3-coder-30b-a3b-instruct |  40 | 0.826 ±0.044  | 0.150 ±0.112          | 0.875 ±0.029            | 48.000 ±0.000  | -0.650 ±0.277      |
| pc_skeleton (no LLM)  | none                         |  40 | 0.828 ±0.036  | 0.100 ±0.094          | 0.863 ±0.027            | 39.400 ±3.655  | -0.800 ±0.188      |
| random edits (no LLM) | none                         |  40 | 0.793 ±0.051  | 0.125 ±0.104          | 0.869 ±0.026            | 48.000 ±0.000  | -0.725 ±0.233      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.824 ±0.042  | 0.805 ±0.063   | 0.871 ±0.026  | 2.050 ±0.460 | 2.150 ±0.205         | 0.960 ±0.027       | 0.109 ±0.065         |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.826 ±0.044  | 0.812 ±0.062   | 0.884 ±0.023  | 1.925 ±0.457 | 2.325 ±0.191         | 0.882 ±0.058       | 0.300 ±0.136         |

## Table 4 — scaling with graph size

| arm                | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.848 ±0.061  | 0.250 ±0.195          |
| probe              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.800 ±0.056  | 0.050 ±0.098          |
| probe              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.854 ±0.069  | 0.250 ±0.195          |
| probe              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.799 ±0.056  | 0.050 ±0.098          |
| probe_random_edits | none                         |       1 |  20 | 0.771 ±0.088  | 0.150 ±0.161          |
| probe_random_edits | none                         |       2 |  20 | 0.815 ±0.052  | 0.100 ±0.135          |
| probe_skel_only    | none                         |       1 |  20 | 0.858 ±0.048  | 0.150 ±0.161          |
| probe_skel_only    | none                         |       2 |  20 | 0.797 ±0.051  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm   | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.824 ±0.042  | 1078.600 ±45.639 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7642 |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.826 ±0.044  | 1755.875 ±81.052 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4707 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 2.000 ±0.000    | 1.450 ±0.210 | 0.824 ±0.042  | 0.882 ±0.025            |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 1.975 ±0.049    | 1.950 ±0.068 | 0.826 ±0.044  | 0.875 ±0.029            |
| probe_random_edits | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.793 ±0.051  | 0.869 ±0.026            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   0.963 |
|      2 |   0.396 |
|      3 |   0.252 |
