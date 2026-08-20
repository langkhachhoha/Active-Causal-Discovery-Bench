# Study 2 — PROBE

80 successful episodes from `traces/study2/edits_2`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm               | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens     | completion_tokens   | total_tokens     | cost_usd     | llm_calls    | wall_sec      |
|:------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:-----------------|:-------------|:-------------|:--------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 98.375 ±9.623       | 1079.188 ±74.091 | 0.000 ±0.000 | 1.000 ±0.000 | 2.730 ±0.720  |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 0.815 ±0.101   | 0.829 ±0.063  | 2.188 ±0.805 | 0.781 ±0.102 | 2.188 ±0.367         | 1443.938 ±102.784 | 142.188 ±16.784     | 1586.125 ±98.786 | 0.001 ±0.000 | 1.000 ±0.000 | 10.422 ±4.527 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 98.375 ±9.623       | 1079.188 ±74.091 | 0.000 ±0.000 | 1.000 ±0.000 | 1.778 ±0.932  |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 0.815 ±0.101   | 0.829 ±0.063  | 2.188 ±0.805 | 0.781 ±0.102 | 2.188 ±0.367         | 1443.938 ±102.784 | 142.188 ±16.784     | 1586.125 ±98.786 | 0.001 ±0.000 | 1.000 ±0.000 | 8.891 ±4.663  |
| probe_skel_only   | none                         |  16 | 0.775 ±0.077  | 0.753 ±0.115   | 0.816 ±0.060  | 2.438 ±0.737 | 0.875 ±0.097 | 1.875 ±0.352         | —                 | —                   | —                | —            | —            | 0.686 ±0.644  |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source    | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:---------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair           | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.125 ±0.167          | 0.854 ±0.049            | 47.250 ±1.470  | -0.750 ±0.335      |
| llm_repair           | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 0.188 ±0.198          | 0.833 ±0.063            | 43.500 ±5.350  | -0.625 ±0.395      |
| llm_repair + pc_mec  | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.125 ±0.167          | 0.854 ±0.049            | 47.250 ±1.470  | -0.750 ±0.335      |
| llm_repair + pc_mec  | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 0.188 ±0.198          | 0.833 ±0.063            | 43.500 ±5.350  | -0.625 ±0.395      |
| pc_skeleton (no LLM) | none                         |  16 | 0.775 ±0.077  | 0.125 ±0.167          | 0.816 ±0.060            | 33.000 ±7.692  | -0.750 ±0.335      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 0.782 ±0.103   | 0.835 ±0.062  | 2.312 ±0.959 | 2.125 ±0.395         | 0.941 ±0.062       | 0.169 ±0.176         |
| probe | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 0.815 ±0.101   | 0.829 ±0.063  | 2.188 ±0.805 | 2.188 ±0.367         | 0.927 ±0.073       | 0.199 ±0.190         |

## Table 4 — scaling with graph size

| arm               | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe             | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.897 ±0.056  | 0.250 ±0.321          |
| probe             | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe             | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.870 ±0.081  | 0.250 ±0.321          |
| probe             | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.746 ±0.127  | 0.125 ±0.245          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.897 ±0.056  | 0.250 ±0.321          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.870 ±0.081  | 0.250 ±0.321          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.746 ±0.127  | 0.125 ±0.245          |
| probe_skel_only   | none                         |       1 |   8 | 0.840 ±0.101  | 0.250 ±0.321          |
| probe_skel_only   | none                         |       2 |   8 | 0.710 ±0.103  | 0.000 ±0.000          |

## Table 5 — quality per token

| arm               | model_tag                    |   n | directed_f1   | total_tokens     | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------------------|:-----------------------------|----:|:--------------|:-----------------|:-------------|:-------------|-------------------:|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 1079.188 ±74.091 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7416 |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 1586.125 ±98.786 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5097 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.800 ±0.087  | 1079.188 ±74.091 | 0.000 ±0.000 | 1.000 ±0.000 |             0.7416 |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.808 ±0.079  | 1586.125 ±98.786 | 0.001 ±0.000 | 1.000 ±0.000 |             0.5097 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm               | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 2.000 ±0.000    | 1.562 ±0.399 | 0.800 ±0.087  | 0.854 ±0.049            |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.625 ±0.469    | 1.375 ±0.469 | 0.808 ±0.079  | 0.833 ±0.063            |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 2.000 ±0.000    | 1.562 ±0.399 | 0.800 ±0.087  | 0.854 ±0.049            |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.625 ±0.469    | 1.375 ±0.469 | 0.808 ±0.079  | 0.833 ±0.063            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   1.052 |
|      2 |   0.524 |
|      3 |   0.261 |
