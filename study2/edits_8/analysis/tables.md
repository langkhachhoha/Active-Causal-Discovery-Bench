# Study 2 — PROBE

80 successful episodes from `traces/study2/edits_8`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm               | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens     | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec      |
|:------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:------------------|:-------------|:-------------|:--------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 0.782 ±0.103   | 0.842 ±0.062  | 2.250 ±0.972 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 92.312 ±9.151       | 1073.125 ±74.621  | 0.000 ±0.000 | 1.000 ±0.000 | 2.600 ±0.780  |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 0.797 ±0.106   | 0.825 ±0.057  | 2.375 ±0.837 | 0.792 ±0.110 | 2.125 ±0.303         | 1443.938 ±102.784 | 157.625 ±24.694     | 1601.562 ±104.057 | 0.001 ±0.000 | 1.000 ±0.000 | 14.415 ±7.609 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 0.782 ±0.103   | 0.842 ±0.062  | 2.250 ±0.972 | 0.750 ±0.103 | 2.125 ±0.395         | 980.812 ±75.881   | 92.312 ±9.151       | 1073.125 ±74.621  | 0.000 ±0.000 | 1.000 ±0.000 | 1.409 ±1.018  |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 0.797 ±0.106   | 0.825 ±0.057  | 2.375 ±0.837 | 0.792 ±0.110 | 2.125 ±0.303         | 1443.938 ±102.784 | 157.625 ±24.694     | 1601.562 ±104.057 | 0.001 ±0.000 | 1.000 ±0.000 | 12.393 ±6.867 |
| probe_skel_only   | none                         |  16 | 0.775 ±0.077  | 0.753 ±0.115   | 0.816 ±0.060  | 2.438 ±0.737 | 0.875 ±0.097 | 1.875 ±0.352         | —                 | —                   | —                 | —            | —            | 0.690 ±0.684  |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source    | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:---------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair           | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 0.125 ±0.167          | 0.860 ±0.049            | 48.000 ±0.000  | -0.750 ±0.335      |
| llm_repair           | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 0.188 ±0.198          | 0.840 ±0.051            | 47.000 ±1.960  | -0.625 ±0.395      |
| llm_repair + pc_mec  | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 0.125 ±0.167          | 0.860 ±0.049            | 48.000 ±0.000  | -0.750 ±0.335      |
| llm_repair + pc_mec  | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 0.188 ±0.198          | 0.840 ±0.051            | 47.000 ±1.960  | -0.625 ±0.395      |
| pc_skeleton (no LLM) | none                         |  16 | 0.775 ±0.077  | 0.125 ±0.167          | 0.816 ±0.060            | 33.000 ±7.692  | -0.750 ±0.335      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 0.782 ±0.103   | 0.842 ±0.062  | 2.250 ±0.972 | 2.125 ±0.395         | 0.955 ±0.062       | 0.111 ±0.144         |
| probe | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 0.797 ±0.106   | 0.825 ±0.057  | 2.375 ±0.837 | 2.125 ±0.303         | 0.919 ±0.088       | 0.177 ±0.189         |

## Table 4 — scaling with graph size

| arm               | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| probe             | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.910 ±0.049  | 0.250 ±0.321          |
| probe             | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe             | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.818 ±0.133  | 0.250 ±0.321          |
| probe             | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.778 ±0.080  | 0.125 ±0.245          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       1 |   8 | 0.910 ±0.049  | 0.250 ±0.321          |
| probe_repair_only | gpt-4o-mini-2024-07-18       |       2 |   8 | 0.704 ±0.140  | 0.000 ±0.000          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       1 |   8 | 0.818 ±0.133  | 0.250 ±0.321          |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |       2 |   8 | 0.778 ±0.080  | 0.125 ±0.245          |
| probe_skel_only   | none                         |       1 |   8 | 0.840 ±0.101  | 0.250 ±0.321          |
| probe_skel_only   | none                         |       2 |   8 | 0.710 ±0.103  | 0.000 ±0.000          |

## Table 5 — quality per token

| arm               | model_tag                    |   n | directed_f1   | total_tokens      | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------------------|:-----------------------------|----:|:--------------|:------------------|:-------------|:-------------|-------------------:|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 1073.125 ±74.621  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7522 |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 1601.562 ±104.057 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4985 |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 0.807 ±0.088  | 1073.125 ±74.621  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7522 |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.798 ±0.076  | 1601.562 ±104.057 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4985 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm               | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe             | gpt-4o-mini-2024-07-18       |  16 | 3.562 ±0.356    | 3.062 ±0.704 | 0.807 ±0.088  | 0.860 ±0.049            |
| probe             | qwen3-coder-30b-a3b-instruct |  16 | 0.625 ±0.756    | 2.688 ±0.992 | 0.798 ±0.076  | 0.840 ±0.051            |
| probe_repair_only | gpt-4o-mini-2024-07-18       |  16 | 3.562 ±0.356    | 3.062 ±0.704 | 0.807 ±0.088  | 0.860 ±0.049            |
| probe_repair_only | qwen3-coder-30b-a3b-instruct |  16 | 0.625 ±0.756    | 2.688 ±0.992 | 0.798 ±0.076  | 0.840 ±0.051            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   0.966 |
|      2 |   0.378 |
|      3 |   0.176 |
