# Study 2 — PROBE

360 successful episodes from `study2/reserve_n60`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                          | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec     |
|:-----------------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:-----------------|:-------------|:-------------|:-------------|
| probe                        | gpt-4o-mini-2024-07-18       |  40 | 0.818 ±0.042  | 0.785 ±0.078   | 0.872 ±0.024  | 2.100 ±0.459 | 0.746 ±0.065 | 2.225 ±0.204         | 979.825 ±46.933  | 96.375 ±5.454       | 1076.200 ±45.979 | 0.000 ±0.000 | 1.000 ±0.000 | 3.703 ±0.413 |
| probe                        | qwen3-coder-30b-a3b-instruct |  40 | 0.807 ±0.052  | 0.788 ±0.077   | 0.875 ±0.026  | 2.075 ±0.483 | 0.742 ±0.066 | 2.275 ±0.172         | 1461.550 ±63.356 | 321.650 ±71.172     | 1783.200 ±99.108 | 0.001 ±0.000 | 1.000 ±0.000 | 4.899 ±0.651 |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |  40 | 0.804 ±0.038  | 0.804 ±0.075   | 0.844 ±0.025  | 2.475 ±0.455 | 0.754 ±0.063 | 2.275 ±0.210         | 979.825 ±46.933  | 96.375 ±5.454       | 1076.200 ±45.979 | 0.000 ±0.000 | 1.000 ±0.000 | 2.965 ±0.476 |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |  40 | 0.793 ±0.047  | 0.807 ±0.073   | 0.850 ±0.026  | 2.450 ±0.471 | 0.742 ±0.063 | 2.275 ±0.210         | 1461.550 ±63.356 | 321.650 ±71.172     | 1783.200 ±99.108 | 0.001 ±0.000 | 1.000 ±0.000 | 3.946 ±0.700 |
| probe_oracle_edits           | none                         |  40 | 0.983 ±0.019  | 0.984 ±0.023   | 0.990 ±0.009  | 0.175 ±0.184 | 0.896 ±0.062 | 1.875 ±0.144         | —                | —                   | —                | —            | —            | 0.768 ±0.124 |
| probe_oracle_edits_noreserve | none                         |  40 | 0.991 ±0.014  | 0.984 ±0.023   | 0.997 ±0.006  | 0.100 ±0.154 | 0.875 ±0.065 | 1.950 ±0.139         | —                | —                   | —                | —            | —            | 0.668 ±0.108 |
| probe_random_edits           | none                         |  40 | 0.793 ±0.051  | 0.798 ±0.069   | 0.864 ±0.026  | 2.200 ±0.450 | 0.712 ±0.060 | 2.325 ±0.215         | —                | —                   | —                | —            | —            | 0.959 ±0.306 |
| probe_random_edits_noreserve | none                         |  40 | 0.765 ±0.042  | 0.786 ±0.079   | 0.819 ±0.025  | 2.775 ±0.394 | 0.708 ±0.057 | 2.350 ±0.228         | —                | —                   | —                | —            | —            | 0.946 ±0.182 |
| probe_skel_only              | none                         |  40 | 0.828 ±0.036  | 0.804 ±0.064   | 0.873 ±0.023  | 1.925 ±0.367 | 0.829 ±0.068 | 1.975 ±0.192         | —                | —                   | —                | —            | —            | 0.733 ±0.308 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source      | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:-----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec    | gpt-4o-mini-2024-07-18       |  40 | 0.818 ±0.042  | 0.175 ±0.119          | 0.884 ±0.027            | 48.000 ±0.000  | -0.600 ±0.279      |
| llm_repair + pc_mec    | qwen3-coder-30b-a3b-instruct |  40 | 0.807 ±0.052  | 0.175 ±0.119          | 0.881 ±0.030            | 48.000 ±0.000  | -0.525 ±0.358      |
| llm_repair, no guard   | gpt-4o-mini-2024-07-18       |  40 | 0.804 ±0.038  | 0.075 ±0.083          | 0.838 ±0.029            | 48.000 ±0.000  | -0.850 ±0.165      |
| llm_repair, no guard   | qwen3-coder-30b-a3b-instruct |  40 | 0.793 ±0.047  | 0.075 ±0.083          | 0.835 ±0.032            | 48.000 ±0.000  | -0.850 ±0.165      |
| oracle edits (no LLM)  | none                         |  40 | 0.983 ±0.019  | 0.975 ±0.049          | 0.998 ±0.004            | 48.000 ±0.000  | 1.050 ±0.156       |
| oracle edits, no guard | none                         |  40 | 0.991 ±0.014  | 1.000 ±0.000          | 1.000 ±0.000            | 48.000 ±0.000  | 1.050 ±0.068       |
| pc_skeleton (no LLM)   | none                         |  40 | 0.828 ±0.036  | 0.100 ±0.094          | 0.863 ±0.027            | 39.400 ±3.655  | -0.800 ±0.188      |
| random edits (no LLM)  | none                         |  40 | 0.793 ±0.051  | 0.125 ±0.104          | 0.869 ±0.026            | 48.000 ±0.000  | -0.725 ±0.233      |
| random edits, no guard | none                         |  40 | 0.765 ±0.042  | 0.025 ±0.049          | 0.812 ±0.025            | 48.000 ±0.000  | -0.950 ±0.098      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm             | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:----------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe           | gpt-4o-mini-2024-07-18       |  40 | 0.818 ±0.042  | 0.785 ±0.078   | 0.872 ±0.024  | 2.100 ±0.459 | 2.225 ±0.204         | 0.964 ±0.028       | 0.096 ±0.062         |
| probe           | qwen3-coder-30b-a3b-instruct |  40 | 0.807 ±0.052  | 0.788 ±0.077   | 0.875 ±0.026  | 2.075 ±0.483 | 2.275 ±0.172         | 0.911 ±0.056       | 0.223 ±0.121         |
| probe_noreserve | gpt-4o-mini-2024-07-18       |  40 | 0.804 ±0.038  | 0.804 ±0.075   | 0.844 ±0.025  | 2.475 ±0.455 | 2.275 ±0.210         | 0.920 ±0.054       | 0.148 ±0.093         |
| probe_noreserve | qwen3-coder-30b-a3b-instruct |  40 | 0.793 ±0.047  | 0.807 ±0.073   | 0.850 ±0.026  | 2.450 ±0.471 | 2.275 ±0.210         | 0.893 ±0.059       | 0.212 ±0.118         |

## Table 4 — scaling with graph size

| arm                          | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-----------------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe                        | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.839 ±0.062  | 0.300 ±0.206          |
| probe                        | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.796 ±0.057  | 0.050 ±0.098          |
| probe                        | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.796 ±0.085  | 0.300 ±0.206          |
| probe                        | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.818 ±0.061  | 0.050 ±0.098          |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.840 ±0.048  | 0.150 ±0.161          |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.768 ±0.054  | 0.000 ±0.000          |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.794 ±0.075  | 0.150 ±0.161          |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.791 ±0.057  | 0.000 ±0.000          |
| probe_oracle_edits           | none                         |       1 |  20 | 0.978 ±0.028  | 0.950 ±0.098          |
| probe_oracle_edits           | none                         |       2 |  20 | 0.988 ±0.024  | 1.000 ±0.000          |
| probe_oracle_edits_noreserve | none                         |       1 |  20 | 1.000 ±0.000  | 1.000 ±0.000          |
| probe_oracle_edits_noreserve | none                         |       2 |  20 | 0.981 ±0.027  | 1.000 ±0.000          |
| probe_random_edits           | none                         |       1 |  20 | 0.771 ±0.088  | 0.150 ±0.161          |
| probe_random_edits           | none                         |       2 |  20 | 0.815 ±0.052  | 0.100 ±0.135          |
| probe_random_edits_noreserve | none                         |       1 |  20 | 0.748 ±0.071  | 0.000 ±0.000          |
| probe_random_edits_noreserve | none                         |       2 |  20 | 0.782 ±0.046  | 0.050 ±0.098          |
| probe_skel_only              | none                         |       1 |  20 | 0.858 ±0.048  | 0.150 ±0.161          |
| probe_skel_only              | none                         |       2 |  20 | 0.797 ±0.051  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm             | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:----------------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe           | gpt-4o-mini-2024-07-18       |  40 | 0.818 ±0.042  | 1076.200 ±45.979 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7599 |
| probe           | qwen3-coder-30b-a3b-instruct |  40 | 0.807 ±0.052  | 1783.200 ±99.108 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4524 |
| probe_noreserve | gpt-4o-mini-2024-07-18       |  40 | 0.804 ±0.038  | 1076.200 ±45.979 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7469 |
| probe_noreserve | qwen3-coder-30b-a3b-instruct |  40 | 0.793 ±0.047  | 1783.200 ±99.108 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4446 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                          | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-----------------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe                        | gpt-4o-mini-2024-07-18       |  40 | 3.575 ±0.170    | 2.925 ±0.347 | 0.818 ±0.042  | 0.884 ±0.027            |
| probe                        | qwen3-coder-30b-a3b-instruct |  40 | 3.200 ±0.299    | 3.450 ±0.262 | 0.807 ±0.052  | 0.881 ±0.030            |
| probe_noreserve              | gpt-4o-mini-2024-07-18       |  40 | 3.575 ±0.170    | 2.925 ±0.347 | 0.804 ±0.038  | 0.838 ±0.029            |
| probe_noreserve              | qwen3-coder-30b-a3b-instruct |  40 | 3.200 ±0.299    | 3.450 ±0.262 | 0.793 ±0.047  | 0.835 ±0.032            |
| probe_oracle_edits           | none                         |  40 | 0.225 ±0.131    | 1.400 ±0.241 | 0.983 ±0.019  | 0.998 ±0.004            |
| probe_oracle_edits_noreserve | none                         |  40 | 0.225 ±0.131    | 1.400 ±0.241 | 0.991 ±0.014  | 1.000 ±0.000            |
| probe_random_edits           | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.793 ±0.051  | 0.869 ±0.026            |
| probe_random_edits_noreserve | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.765 ±0.042  | 0.812 ±0.025            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   0.996 |
|      2 |   0.341 |
|      3 |   0.155 |
