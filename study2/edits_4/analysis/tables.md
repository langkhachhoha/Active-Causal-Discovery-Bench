# Study 2 — PROBE

80 successful episodes from `traces/study2/edits_4`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm               | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens     | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec      |
|:------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:------------------|:-------------|:-------------|:--------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 100.188 ±8.775      | 1081.000 ±75.863  | 0.000 ±0.000 | 1.000 ±0.000 | 2.844 ±0.725  |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 0.815 ±0.101   | 0.832 ±0.055  | 2.125 ±0.690 | 0.708 ±0.105 | 2.250 ±0.283         | 1443.938 ±102.784 | 149.312 ±25.522     | 1593.250 ±106.733 | 0.001 ±0.000 | 1.000 ±0.000 | 10.628 ±4.647 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 100.188 ±8.775      | 1081.000 ±75.863  | 0.000 ±0.000 | 1.000 ±0.000 | 1.578 ±0.984  |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 0.815 ±0.101   | 0.832 ±0.055  | 2.125 ±0.690 | 0.708 ±0.105 | 2.250 ±0.283         | 1443.938 ±102.784 | 149.312 ±25.522     | 1593.250 ±106.733 | 0.001 ±0.000 | 1.000 ±0.000 | 8.510 ±4.851  |
| probe_skel_only   | none                         |  16 | 0.775 ±0.077  | 0.753 ±0.115   | 0.816 ±0.060  | 2.438 ±0.737 | 0.875 ±0.097 | 1.875 ±0.352         | —                 | —                   | —                 | —            | —            | 0.718 ±0.690  |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source    | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:---------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair           | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.125 ±0.167          | 0.854 ±0.049            | 48.000 ±0.000  | -0.750 ±0.335      |
| llm_repair           | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 0.188 ±0.198          | 0.847 ±0.052            | 47.000 ±1.960  | -0.625 ±0.395      |
| llm_repair + pc_mec  | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.125 ±0.167          | 0.854 ±0.049            | 48.000 ±0.000  | -0.750 ±0.335      |
| llm_repair + pc_mec  | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 0.188 ±0.198          | 0.847 ±0.052            | 47.000 ±1.960  | -0.625 ±0.395      |
| pc_skeleton (no LLM) | none                         |  16 | 0.775 ±0.077  | 0.125 ±0.167          | 0.816 ±0.060            | 33.000 ±7.692  | -0.750 ±0.335      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 2.125 ±0.395         | 0.975 ±0.037       | 0.082 ±0.114         |
| probe | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 0.815 ±0.101   | 0.832 ±0.055  | 2.125 ±0.690 | 2.250 ±0.283         | 0.914 ±0.087       | 0.195 ±0.189         |

## Table 4 — scaling with graph size

| arm               | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe             | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.897 ±0.056  | 0.250 ±0.321          |
| probe             | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe             | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.869 ±0.077  | 0.250 ±0.321          |
| probe             | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.778 ±0.080  | 0.125 ±0.245          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.897 ±0.056  | 0.250 ±0.321          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.869 ±0.077  | 0.250 ±0.321          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.778 ±0.080  | 0.125 ±0.245          |
| probe_skel_only   | none                         |       1 |   8 | 0.840 ±0.101  | 0.250 ±0.321          |
| probe_skel_only   | none                         |       2 |   8 | 0.710 ±0.103  | 0.000 ±0.000          |

## Table 5 — quality per token

| arm               | model_tag                    |   n | directed_f1   | total_tokens      | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------------------|:-----------------------------|----:|:--------------|:------------------|:-------------|:-------------|-------------------:|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 1081.000 ±75.863  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7404 |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 1593.250 ±106.733 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5169 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 1081.000 ±75.863  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7404 |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.824 ±0.058  | 1593.250 ±106.733 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5169 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm               | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 3.500 ±0.310    | 2.938 ±0.704 | 0.800 ±0.087  | 0.854 ±0.049            |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.938 ±0.789    | 2.500 ±0.645 | 0.824 ±0.058  | 0.847 ±0.052            |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 3.500 ±0.310    | 2.938 ±0.704 | 0.800 ±0.087  | 0.854 ±0.049            |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.938 ±0.789    | 2.500 ±0.645 | 0.824 ±0.058  | 0.847 ±0.052            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   1.036 |
|      2 |   0.401 |
|      3 |   0.195 |
