# Study 2 — PROBE

360 successful episodes from `study2/reserve_n300`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                          | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec     |
|:-----------------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:-----------------|:-------------|:-------------|:-------------|
| probe                        | gpt-4o-mini-2024-07-18       |  40 | 0.957 ±0.024  | 0.957 ±0.037   | 0.965 ±0.017  | 0.525 ±0.272 | 0.692 ±0.057 | 2.450 ±0.171         | 979.275 ±46.871  | 100.675 ±5.518      | 1079.950 ±46.419 | 0.000 ±0.000 | 1.000 ±0.000 | 4.051 ±0.410 |
| probe                        | qwen3-coder-30b-a3b-instruct |  40 | 0.940 ±0.033  | 0.934 ±0.050   | 0.961 ±0.017  | 0.650 ±0.310 | 0.646 ±0.044 | 2.575 ±0.155         | 1463.550 ±63.445 | 286.025 ±23.430     | 1749.575 ±72.438 | 0.001 ±0.000 | 1.000 ±0.000 | 5.035 ±0.574 |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |  40 | 0.904 ±0.024  | 0.954 ±0.037   | 0.911 ±0.017  | 1.325 ±0.301 | 0.717 ±0.059 | 2.375 ±0.207         | 979.275 ±46.871  | 100.675 ±5.518      | 1079.950 ±46.419 | 0.000 ±0.000 | 1.000 ±0.000 | 3.270 ±0.434 |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |  40 | 0.877 ±0.033  | 0.926 ±0.052   | 0.901 ±0.017  | 1.575 ±0.313 | 0.671 ±0.049 | 2.525 ±0.186         | 1463.550 ±63.445 | 286.025 ±23.430     | 1749.575 ±72.438 | 0.001 ±0.000 | 1.000 ±0.000 | 4.030 ±0.579 |
| probe_oracle_edits           | none                         |  40 | 0.991 ±0.014  | 1.000 ±0.000   | 0.996 ±0.006  | 0.075 ±0.108 | 0.900 ±0.059 | 1.900 ±0.154         | —                | —                   | —                | —            | —            | 0.911 ±0.168 |
| probe_oracle_edits_noreserve | none                         |  40 | 0.998 ±0.004  | 1.000 ±0.000   | 0.998 ±0.004  | 0.025 ±0.049 | 0.892 ±0.060 | 1.925 ±0.163         | —                | —                   | —                | —            | —            | 0.827 ±0.194 |
| probe_random_edits           | none                         |  40 | 0.919 ±0.040  | 0.927 ±0.054   | 0.955 ±0.018  | 0.850 ±0.381 | 0.646 ±0.044 | 2.550 ±0.171         | —                | —                   | —                | —            | —            | 1.208 ±0.334 |
| probe_random_edits_noreserve | none                         |  40 | 0.868 ±0.033  | 0.930 ±0.051   | 0.894 ±0.016  | 1.700 ±0.331 | 0.658 ±0.047 | 2.550 ±0.171         | —                | —                   | —                | —            | —            | 1.082 ±0.190 |
| probe_skel_only              | none                         |  40 | 0.945 ±0.026  | 0.942 ±0.046   | 0.957 ±0.018  | 0.650 ±0.286 | 0.900 ±0.059 | 1.875 ±0.160         | —                | —                   | —                | —            | —            | 0.813 ±0.317 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source      | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:-----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec    | gpt-4o-mini-2024-07-18       |  40 | 0.957 ±0.024  | 0.675 ±0.147          | 0.963 ±0.022            | 48.000 ±0.000  | 0.375 ±0.303       |
| llm_repair + pc_mec    | qwen3-coder-30b-a3b-instruct |  40 | 0.940 ±0.033  | 0.625 ±0.152          | 0.957 ±0.023            | 48.000 ±0.000  | 0.275 ±0.314       |
| llm_repair, no guard   | gpt-4o-mini-2024-07-18       |  40 | 0.904 ±0.024  | 0.100 ±0.094          | 0.907 ±0.022            | 48.000 ±0.000  | -0.800 ±0.188      |
| llm_repair, no guard   | qwen3-coder-30b-a3b-instruct |  40 | 0.877 ±0.033  | 0.050 ±0.068          | 0.900 ±0.022            | 48.000 ±0.000  | -0.850 ±0.217      |
| oracle edits (no LLM)  | none                         |  40 | 0.991 ±0.014  | 0.975 ±0.049          | 0.998 ±0.004            | 48.000 ±0.000  | 0.975 ±0.111       |
| oracle edits, no guard | none                         |  40 | 0.998 ±0.004  | 1.000 ±0.000          | 1.000 ±0.000            | 48.000 ±0.000  | 1.025 ±0.049       |
| pc_skeleton (no LLM)   | none                         |  40 | 0.945 ±0.026  | 0.575 ±0.155          | 0.954 ±0.022            | 46.000 ±1.661  | 0.150 ±0.310       |
| random edits (no LLM)  | none                         |  40 | 0.919 ±0.040  | 0.575 ±0.155          | 0.951 ±0.024            | 48.000 ±0.000  | 0.200 ±0.331       |
| random edits, no guard | none                         |  40 | 0.868 ±0.033  | 0.000 ±0.000          | 0.887 ±0.021            | 48.000 ±0.000  | -1.000 ±0.000      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm             | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:----------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe           | gpt-4o-mini-2024-07-18       |  40 | 0.957 ±0.024  | 0.957 ±0.037   | 0.965 ±0.017  | 0.525 ±0.272 | 2.450 ±0.171         | 0.924 ±0.039       | 0.210 ±0.087         |
| probe           | qwen3-coder-30b-a3b-instruct |  40 | 0.940 ±0.033  | 0.934 ±0.050   | 0.961 ±0.017  | 0.650 ±0.310 | 2.575 ±0.155         | 0.898 ±0.048       | 0.309 ±0.117         |
| probe_noreserve | gpt-4o-mini-2024-07-18       |  40 | 0.904 ±0.024  | 0.954 ±0.037   | 0.911 ±0.017  | 1.325 ±0.301 | 2.375 ±0.207         | 0.894 ±0.059       | 0.202 ±0.110         |
| probe_noreserve | qwen3-coder-30b-a3b-instruct |  40 | 0.877 ±0.033  | 0.926 ±0.052   | 0.901 ±0.017  | 1.575 ±0.313 | 2.525 ±0.186         | 0.775 ±0.072       | 0.476 ±0.146         |

## Table 4 — scaling with graph size

| arm                          | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-----------------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe                        | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.957 ±0.034  | 0.750 ±0.195          |
| probe                        | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.957 ±0.034  | 0.600 ±0.220          |
| probe                        | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.926 ±0.058  | 0.700 ±0.206          |
| probe                        | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.954 ±0.034  | 0.550 ±0.224          |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.901 ±0.035  | 0.100 ±0.135          |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.908 ±0.035  | 0.100 ±0.135          |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.846 ±0.054  | 0.050 ±0.098          |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.908 ±0.033  | 0.050 ±0.098          |
| probe_oracle_edits           | none                         |       1 |  20 | 0.983 ±0.027  | 0.950 ±0.098          |
| probe_oracle_edits           | none                         |       2 |  20 | 1.000 ±0.000  | 1.000 ±0.000          |
| probe_oracle_edits_noreserve | none                         |       1 |  20 | 0.996 ±0.008  | 1.000 ±0.000          |
| probe_oracle_edits_noreserve | none                         |       2 |  20 | 1.000 ±0.000  | 1.000 ±0.000          |
| probe_random_edits           | none                         |       1 |  20 | 0.920 ±0.061  | 0.650 ±0.214          |
| probe_random_edits           | none                         |       2 |  20 | 0.918 ±0.054  | 0.500 ±0.225          |
| probe_random_edits_noreserve | none                         |       1 |  20 | 0.859 ±0.046  | 0.000 ±0.000          |
| probe_random_edits_noreserve | none                         |       2 |  20 | 0.877 ±0.048  | 0.000 ±0.000          |
| probe_skel_only              | none                         |       1 |  20 | 0.942 ±0.041  | 0.650 ±0.214          |
| probe_skel_only              | none                         |       2 |  20 | 0.947 ±0.035  | 0.500 ±0.225          |

## Table 5 — quality per token

| arm             | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:----------------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe           | gpt-4o-mini-2024-07-18       |  40 | 0.957 ±0.024  | 1079.950 ±46.419 | 0.000 ±0.000 | 1.000 ±0.000 |             0.8861 |
| probe           | qwen3-coder-30b-a3b-instruct |  40 | 0.940 ±0.033  | 1749.575 ±72.438 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5372 |
| probe_noreserve | gpt-4o-mini-2024-07-18       |  40 | 0.904 ±0.024  | 1079.950 ±46.419 | 0.000 ±0.000 | 1.000 ±0.000 |             0.8373 |
| probe_noreserve | qwen3-coder-30b-a3b-instruct |  40 | 0.877 ±0.033  | 1749.575 ±72.438 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5014 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                          | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-----------------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe                        | gpt-4o-mini-2024-07-18       |  40 | 3.825 ±0.119    | 2.625 ±0.382 | 0.957 ±0.024  | 0.963 ±0.022            |
| probe                        | qwen3-coder-30b-a3b-instruct |  40 | 2.825 ±0.402    | 3.325 ±0.317 | 0.940 ±0.033  | 0.957 ±0.023            |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |  40 | 3.825 ±0.119    | 2.625 ±0.382 | 0.904 ±0.024  | 0.907 ±0.022            |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |  40 | 2.825 ±0.402    | 3.325 ±0.317 | 0.877 ±0.033  | 0.900 ±0.022            |
| probe_oracle_edits           | none                         |  40 | 0.125 ±0.104    | 0.450 ±0.185 | 0.991 ±0.014  | 0.998 ±0.004            |
| probe_oracle_edits_noreserve | none                         |  40 | 0.125 ±0.104    | 0.450 ±0.185 | 0.998 ±0.004  | 1.000 ±0.000            |
| probe_random_edits           | none                         |  40 | 4.000 ±0.000    | 4.000 ±0.000 | 0.919 ±0.040  | 0.951 ±0.024            |
| probe_random_edits_noreserve | none                         |  40 | 4.000 ±0.000    | 4.000 ±0.000 | 0.868 ±0.033  | 0.887 ±0.021            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   1.137 |
|      2 |   0.394 |
|      3 |   0.264 |
