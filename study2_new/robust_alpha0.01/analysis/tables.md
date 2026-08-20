# Study 2 — PROBE

280 successful episodes from `study2/robust_alpha0.01`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec     |
|:-------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:-----------------|:-------------|:-------------|:-------------|
| oracle             | none                         |  40 | 1.000 ±0.000  | 1.000 ±0.000   | 1.000 ±0.000  | 0.000 ±0.000 | 1.000 ±0.000 | 0.000 ±0.000         | —                | —                   | —                | —            | —            | 0.005 ±0.002 |
| pc_greedy_meek     | none                         |  40 | 0.702 ±0.045  | 0.656 ±0.084   | 0.810 ±0.030  | 2.900 ±0.408 | 0.875 ±0.065 | 1.650 ±0.217         | —                | —                   | —                | —            | —            | 0.206 ±0.287 |
| probe              | gpt-4o-mini-2024-07-18       |  40 | 0.769 ±0.051  | 0.744 ±0.074   | 0.822 ±0.031  | 2.525 ±0.496 | 0.758 ±0.065 | 2.150 ±0.228         | 979.825 ±46.933  | 99.750 ±5.024       | 1079.575 ±48.036 | 0.000 ±0.000 | 1.000 ±0.000 | 2.870 ±0.288 |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 0.775 ±0.051  | 0.760 ±0.076   | 0.831 ±0.031  | 2.450 ±0.506 | 0.796 ±0.067 | 2.050 ±0.221         | 1461.550 ±63.356 | 289.925 ±32.686     | 1751.475 ±80.515 | 0.001 ±0.000 | 1.000 ±0.000 | 3.901 ±0.445 |
| probe_oracle_edits | none                         |  40 | 0.970 ±0.028  | 0.965 ±0.038   | 0.981 ±0.020  | 0.325 ±0.325 | 0.908 ±0.058 | 1.850 ±0.150         | —                | —                   | —                | —            | —            | 0.598 ±0.299 |
| probe_random_edits | none                         |  40 | 0.739 ±0.057  | 0.728 ±0.080   | 0.808 ±0.033  | 2.750 ±0.524 | 0.729 ±0.063 | 2.250 ±0.219         | —                | —                   | —                | —            | —            | 0.795 ±0.306 |
| probe_skel_only    | none                         |  40 | 0.773 ±0.038  | 0.748 ±0.055   | 0.810 ±0.030  | 2.450 ±0.378 | 0.829 ±0.068 | 2.000 ±0.186         | —                | —                   | —                | —            | —            | 0.671 ±0.305 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source     | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec   | gpt-4o-mini-2024-07-18       |  40 | 0.769 ±0.051  | 0.125 ±0.104          | 0.835 ±0.032            | 48.000 ±0.000  | -0.725 ±0.233      |
| llm_repair + pc_mec   | qwen3-coder-30b-a3b-instruct |  40 | 0.775 ±0.051  | 0.100 ±0.094          | 0.837 ±0.032            | 48.000 ±0.000  | -0.750 ±0.251      |
| oracle edits (no LLM) | none                         |  40 | 0.970 ±0.028  | 0.950 ±0.068          | 0.983 ±0.023            | 48.000 ±0.000  | 1.000 ±0.186       |
| pc_skeleton (no LLM)  | none                         |  40 | 0.773 ±0.038  | 0.050 ±0.068          | 0.810 ±0.030            | 31.650 ±4.684  | -0.900 ±0.137      |
| random edits (no LLM) | none                         |  40 | 0.739 ±0.057  | 0.075 ±0.083          | 0.818 ±0.031            | 48.000 ±0.000  | -0.850 ±0.165      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.769 ±0.051  | 0.744 ±0.074   | 0.822 ±0.031  | 2.525 ±0.496 | 2.150 ±0.228         | 0.952 ±0.037       | 0.116 ±0.079         |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.775 ±0.051  | 0.760 ±0.076   | 0.831 ±0.031  | 2.450 ±0.506 | 2.050 ±0.221         | 0.924 ±0.047       | 0.171 ±0.093         |

## Table 4 — scaling with graph size

| arm                | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| oracle             | none                         |       1 |  20 | 1.000 ±0.000  | —                     |
| oracle             | none                         |       2 |  20 | 1.000 ±0.000  | —                     |
| pc_greedy_meek     | none                         |       1 |  20 | 0.703 ±0.059  | —                     |
| pc_greedy_meek     | none                         |       2 |  20 | 0.701 ±0.069  | —                     |
| probe              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.794 ±0.077  | 0.200 ±0.180          |
| probe              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.744 ±0.066  | 0.050 ±0.098          |
| probe              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.807 ±0.065  | 0.150 ±0.161          |
| probe              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.742 ±0.077  | 0.050 ±0.098          |
| probe_oracle_edits | none                         |       1 |  20 | 0.982 ±0.028  | 1.000 ±0.000          |
| probe_oracle_edits | none                         |       2 |  20 | 0.958 ±0.050  | 0.900 ±0.135          |
| probe_random_edits | none                         |       1 |  20 | 0.737 ±0.084  | 0.050 ±0.098          |
| probe_random_edits | none                         |       2 |  20 | 0.741 ±0.079  | 0.100 ±0.135          |
| probe_skel_only    | none                         |       1 |  20 | 0.775 ±0.053  | 0.050 ±0.098          |
| probe_skel_only    | none                         |       2 |  20 | 0.770 ±0.055  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm   | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.769 ±0.051  | 1079.575 ±48.036 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7123 |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.775 ±0.051  | 1751.475 ±80.515 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4422 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 3.475 ±0.172    | 2.875 ±0.379 | 0.769 ±0.051  | 0.835 ±0.032            |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 2.950 ±0.336    | 3.400 ±0.288 | 0.775 ±0.051  | 0.837 ±0.032            |
| probe_oracle_edits | none                         |  40 | 0.125 ±0.104    | 2.100 ±0.296 | 0.970 ±0.028  | 0.983 ±0.023            |
| probe_random_edits | none                         |  40 | 3.925 ±0.083    | 4.000 ±0.000 | 0.739 ±0.057  | 0.818 ±0.031            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   0.965 |
|      2 |   0.384 |
|      3 |   0.21  |
