# Study 2 — PROBE

160 successful episodes from `study2/edits_e4`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec     |
|:-------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:------------------|:-------------|:-------------|:-------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 0.830 ±0.041  | 0.802 ±0.073   | 0.877 ±0.024  | 1.975 ±0.435 | 0.738 ±0.064 | 2.275 ±0.198         | 979.825 ±46.933  | 100.325 ±5.472      | 1080.150 ±46.189  | 0.000 ±0.000 | 1.000 ±0.000 | 3.160 ±0.438 |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 0.838 ±0.043  | 0.841 ±0.056   | 0.886 ±0.025  | 1.825 ±0.449 | 0.733 ±0.065 | 2.300 ±0.175         | 1494.625 ±82.890 | 388.025 ±160.070    | 1882.650 ±223.800 | 0.001 ±0.000 | 1.025 ±0.049 | 5.018 ±1.472 |
| probe_random_edits | none                         |  40 | 0.793 ±0.051  | 0.798 ±0.069   | 0.864 ±0.026  | 2.200 ±0.450 | 0.712 ±0.060 | 2.325 ±0.215         | —                | —                   | —                 | —            | —            | 0.748 ±0.306 |
| probe_skel_only    | none                         |  40 | 0.828 ±0.036  | 0.804 ±0.064   | 0.873 ±0.023  | 1.925 ±0.367 | 0.829 ±0.068 | 1.975 ±0.192         | —                | —                   | —                 | —            | —            | 0.635 ±0.308 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source     | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec   | gpt-4o-mini-2024-07-18       |  40 | 0.830 ±0.041  | 0.150 ±0.112          | 0.886 ±0.026            | 48.000 ±0.000  | -0.675 ±0.247      |
| llm_repair + pc_mec   | qwen3-coder-30b-a3b-instruct |  40 | 0.838 ±0.043  | 0.175 ±0.119          | 0.881 ±0.030            | 48.000 ±0.000  | -0.600 ±0.288      |
| pc_skeleton (no LLM)  | none                         |  40 | 0.828 ±0.036  | 0.100 ±0.094          | 0.863 ±0.027            | 39.400 ±3.655  | -0.800 ±0.188      |
| random edits (no LLM) | none                         |  40 | 0.793 ±0.051  | 0.125 ±0.104          | 0.869 ±0.026            | 48.000 ±0.000  | -0.725 ±0.233      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.830 ±0.041  | 0.802 ±0.073   | 0.877 ±0.024  | 1.975 ±0.435 | 2.275 ±0.198         | 0.939 ±0.039       | 0.149 ±0.080         |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.838 ±0.043  | 0.841 ±0.056   | 0.886 ±0.025  | 1.825 ±0.449 | 2.300 ±0.175         | 0.896 ±0.059       | 0.255 ±0.130         |

## Table 4 — scaling with graph size

| arm                | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.842 ±0.064  | 0.250 ±0.195          |
| probe              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.818 ±0.052  | 0.050 ±0.098          |
| probe              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.865 ±0.059  | 0.300 ±0.206          |
| probe              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.812 ±0.061  | 0.050 ±0.098          |
| probe_random_edits | none                         |       1 |  20 | 0.771 ±0.088  | 0.150 ±0.161          |
| probe_random_edits | none                         |       2 |  20 | 0.815 ±0.052  | 0.100 ±0.135          |
| probe_skel_only    | none                         |       1 |  20 | 0.858 ±0.048  | 0.150 ±0.161          |
| probe_skel_only    | none                         |       2 |  20 | 0.797 ±0.051  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm   | model_tag                    |   n | directed_f1   | total_tokens      | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------|:-----------------------------|----:|:--------------|:------------------|:-------------|:-------------|-------------------:|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.830 ±0.041  | 1080.150 ±46.189  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7681 |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.838 ±0.043  | 1882.650 ±223.800 | 0.001 ±0.000 | 1.025 ±0.049 |             0.4453 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 3.650 ±0.165    | 3.125 ±0.323 | 0.830 ±0.041  | 0.886 ±0.026            |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 3.350 ±0.286    | 3.425 ±0.252 | 0.838 ±0.043  | 0.881 ±0.030            |
| probe_random_edits | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.793 ±0.051  | 0.869 ±0.026            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   1.013 |
|      2 |   0.37  |
|      3 |   0.207 |
