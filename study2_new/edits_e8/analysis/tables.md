# Study 2 — PROBE

160 successful episodes from `study2/edits_e8`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec     |
|:-------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:-----------------|:-------------|:-------------|:-------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 0.819 ±0.040  | 0.777 ±0.076   | 0.870 ±0.023  | 2.075 ±0.429 | 0.721 ±0.061 | 2.300 ±0.213         | 979.825 ±46.933  | 103.325 ±5.127      | 1083.150 ±47.708 | 0.000 ±0.000 | 1.000 ±0.000 | 3.254 ±0.461 |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 0.837 ±0.043  | 0.812 ±0.067   | 0.884 ±0.026  | 1.850 ±0.458 | 0.758 ±0.068 | 2.175 ±0.184         | 1461.550 ±63.356 | 332.350 ±77.612     | 1793.900 ±97.122 | 0.001 ±0.000 | 1.000 ±0.000 | 4.522 ±0.765 |
| probe_random_edits | none                         |  40 | 0.793 ±0.051  | 0.798 ±0.069   | 0.864 ±0.026  | 2.200 ±0.450 | 0.712 ±0.060 | 2.325 ±0.215         | —                | —                   | —                | —            | —            | 0.775 ±0.311 |
| probe_skel_only    | none                         |  40 | 0.828 ±0.036  | 0.804 ±0.064   | 0.873 ±0.023  | 1.925 ±0.367 | 0.829 ±0.068 | 1.975 ±0.192         | —                | —                   | —                | —            | —            | 0.552 ±0.315 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source     | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec   | gpt-4o-mini-2024-07-18       |  40 | 0.819 ±0.040  | 0.100 ±0.094          | 0.877 ±0.025            | 48.000 ±0.000  | -0.800 ±0.188      |
| llm_repair + pc_mec   | qwen3-coder-30b-a3b-instruct |  40 | 0.837 ±0.043  | 0.175 ±0.119          | 0.885 ±0.029            | 48.000 ±0.000  | -0.600 ±0.288      |
| pc_skeleton (no LLM)  | none                         |  40 | 0.828 ±0.036  | 0.100 ±0.094          | 0.863 ±0.027            | 39.400 ±3.655  | -0.800 ±0.188      |
| random edits (no LLM) | none                         |  40 | 0.793 ±0.051  | 0.125 ±0.104          | 0.869 ±0.026            | 48.000 ±0.000  | -0.725 ±0.233      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.819 ±0.040  | 0.777 ±0.076   | 0.870 ±0.023  | 2.075 ±0.429 | 2.300 ±0.213         | 0.934 ±0.041       | 0.160 ±0.087         |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.837 ±0.043  | 0.812 ±0.067   | 0.884 ±0.026  | 1.850 ±0.458 | 2.175 ±0.184         | 0.905 ±0.061       | 0.209 ±0.126         |

## Table 4 — scaling with graph size

| arm                | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.832 ±0.062  | 0.150 ±0.161          |
| probe              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.807 ±0.053  | 0.050 ±0.098          |
| probe              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.859 ±0.063  | 0.300 ±0.206          |
| probe              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.814 ±0.060  | 0.050 ±0.098          |
| probe_random_edits | none                         |       1 |  20 | 0.771 ±0.088  | 0.150 ±0.161          |
| probe_random_edits | none                         |       2 |  20 | 0.815 ±0.052  | 0.100 ±0.135          |
| probe_skel_only    | none                         |       1 |  20 | 0.858 ±0.048  | 0.150 ±0.161          |
| probe_skel_only    | none                         |       2 |  20 | 0.797 ±0.051  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm   | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.819 ±0.040  | 1083.150 ±47.708 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7566 |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.837 ±0.043  | 1793.900 ±97.122 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4664 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 3.625 ±0.167    | 2.975 ±0.333 | 0.819 ±0.040  | 0.877 ±0.025            |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 3.600 ±0.547    | 4.225 ±0.664 | 0.837 ±0.043  | 0.885 ±0.029            |
| probe_random_edits | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.793 ±0.051  | 0.869 ±0.026            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   1     |
|      2 |   0.385 |
|      3 |   0.188 |
