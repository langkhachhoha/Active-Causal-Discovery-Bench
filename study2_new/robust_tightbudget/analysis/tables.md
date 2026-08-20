# Study 2 — PROBE

720 successful episodes from `study2/robust_tightbudget`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm              | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens     | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec     |
|:-----------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:------------------|:-------------|:-------------|:-------------|
| oracle           | none                         |  80 | 1.000 ±0.000  | 1.000 ±0.000   | 1.000 ±0.000  | 0.000 ±0.000 | 1.000 ±0.000 | 0.000 ±0.000         | —                 | —                   | —                 | —            | —            | 0.007 ±0.003 |
| pc_greedy_meek   | none                         |  80 | 0.852 ±0.043  | 0.958 ±0.020   | 0.947 ±0.014  | 1.262 ±0.299 | 1.000 ±0.000 | 1.600 ±0.150         | —                 | —                   | —                 | —            | —            | 0.120 ±0.139 |
| probe            | gpt-4o-mini-2024-07-18       |  80 | 0.913 ±0.027  | 0.951 ±0.025   | 0.960 ±0.014  | 0.900 ±0.280 | 1.000 ±0.000 | 1.700 ±0.132         | 1019.837 ±74.510  | 91.050 ±4.054       | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 | 2.706 ±0.208 |
| probe            | qwen3-coder-30b-a3b-instruct |  80 | 0.922 ±0.028  | 0.964 ±0.019   | 0.960 ±0.014  | 0.850 ±0.298 | 1.000 ±0.000 | 1.712 ±0.131         | 1518.963 ±100.538 | 328.438 ±64.570     | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 | 4.102 ±0.576 |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |  80 | 0.919 ±0.027  | 0.944 ±0.027   | 0.960 ±0.013  | 0.863 ±0.276 | 1.000 ±0.000 | 1.688 ±0.129         | 1019.837 ±74.510  | 91.050 ±4.054       | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 | 1.023 ±0.152 |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.914 ±0.031  | 0.958 ±0.023   | 0.958 ±0.014  | 0.887 ±0.290 | 1.000 ±0.000 | 1.712 ±0.131         | 1518.963 ±100.538 | 328.438 ±64.570     | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 | 2.432 ±0.566 |
| probe_random_sel | gpt-4o-mini-2024-07-18       |  80 | 0.858 ±0.033  | 0.950 ±0.024   | 0.956 ±0.013  | 1.275 ±0.283 | 1.000 ±0.000 | 1.700 ±0.132         | 1019.837 ±74.510  | 91.050 ±4.054       | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 | 1.313 ±0.228 |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.852 ±0.034  | 0.955 ±0.026   | 0.958 ±0.014  | 1.262 ±0.295 | 1.000 ±0.000 | 1.712 ±0.131         | 1518.963 ±100.538 | 328.438 ±64.570     | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 | 3.208 ±0.556 |
| probe_skel_only  | none                         |  80 | 0.874 ±0.040  | 0.948 ±0.025   | 0.947 ±0.014  | 1.087 ±0.306 | 1.000 ±0.000 | 1.650 ±0.135         | —                 | —                   | —                 | —            | —            | 0.619 ±0.093 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source    | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:---------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec  | gpt-4o-mini-2024-07-18       |  80 | 0.913 ±0.027  | 0.613 ±0.107          | 0.960 ±0.014            | 47.575 ±0.511  | 0.338 ±0.251       |
| llm_repair + pc_mec  | qwen3-coder-30b-a3b-instruct |  80 | 0.922 ±0.028  | 0.650 ±0.105          | 0.964 ±0.013            | 46.150 ±1.600  | 0.412 ±0.246       |
| pc_skeleton (no LLM) | none                         |  80 | 0.874 ±0.040  | 0.487 ±0.110          | 0.944 ±0.014            | 37.475 ±3.623  | 0.037 ±0.240       |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm              | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:-----------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe            | gpt-4o-mini-2024-07-18       |  80 | 0.913 ±0.027  | 0.951 ±0.025   | 0.960 ±0.014  | 0.900 ±0.280 | 1.700 ±0.132         | 0.775 ±0.055       | 0.451 ±0.098         |
| probe            | qwen3-coder-30b-a3b-instruct |  80 | 0.922 ±0.028  | 0.964 ±0.019   | 0.960 ±0.014  | 0.850 ±0.298 | 1.712 ±0.131         | 0.750 ±0.058       | 0.551 ±0.122         |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |  80 | 0.919 ±0.027  | 0.944 ±0.027   | 0.960 ±0.013  | 0.863 ±0.276 | 1.688 ±0.129         | 0.786 ±0.055       | 0.446 ±0.105         |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.914 ±0.031  | 0.958 ±0.023   | 0.958 ±0.014  | 0.887 ±0.290 | 1.712 ±0.131         | 0.753 ±0.056       | 0.548 ±0.120         |
| probe_random_sel | gpt-4o-mini-2024-07-18       |  80 | 0.858 ±0.033  | 0.950 ±0.024   | 0.956 ±0.013  | 1.275 ±0.283 | 1.700 ±0.132         | 0.541 ±0.069       | 1.006 ±0.164         |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.852 ±0.034  | 0.955 ±0.026   | 0.958 ±0.014  | 1.262 ±0.295 | 1.712 ±0.131         | 0.537 ±0.068       | 1.051 ±0.165         |

## Table 4 — scaling with graph size

| arm              | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-----------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| oracle           | none                         |       0 |  20 | 1.000 ±0.000  | —                     |
| oracle           | none                         |       1 |  20 | 1.000 ±0.000  | —                     |
| oracle           | none                         |       2 |  20 | 1.000 ±0.000  | —                     |
| oracle           | none                         |       3 |  20 | 1.000 ±0.000  | —                     |
| pc_greedy_meek   | none                         |       0 |  20 | 0.696 ±0.125  | —                     |
| pc_greedy_meek   | none                         |       1 |  20 | 0.917 ±0.047  | —                     |
| pc_greedy_meek   | none                         |       2 |  20 | 0.913 ±0.059  | —                     |
| pc_greedy_meek   | none                         |       3 |  20 | 0.882 ±0.051  | —                     |
| probe            | gpt-4o-mini-2024-07-18       |       0 |  20 | 0.911 ±0.069  | 0.850 ±0.161          |
| probe            | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.929 ±0.052  | 0.700 ±0.206          |
| probe            | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.919 ±0.043  | 0.550 ±0.224          |
| probe            | gpt-4o-mini-2024-07-18       |       3 |  20 | 0.896 ±0.052  | 0.350 ±0.214          |
| probe            | qwen3-coder-30b-a3b-instruct |       0 |  20 | 0.916 ±0.064  | 0.750 ±0.195          |
| probe            | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.973 ±0.031  | 0.900 ±0.135          |
| probe            | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.903 ±0.067  | 0.600 ±0.220          |
| probe            | qwen3-coder-30b-a3b-instruct |       3 |  20 | 0.897 ±0.054  | 0.350 ±0.214          |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |       0 |  20 | 0.911 ±0.069  | 0.850 ±0.161          |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.950 ±0.039  | 0.700 ±0.206          |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.916 ±0.052  | 0.550 ±0.224          |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |       3 |  20 | 0.901 ±0.052  | 0.350 ±0.214          |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |       0 |  20 | 0.883 ±0.082  | 0.750 ±0.195          |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.966 ±0.032  | 0.900 ±0.135          |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.905 ±0.067  | 0.600 ±0.220          |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |       3 |  20 | 0.902 ±0.052  | 0.350 ±0.214          |
| probe_random_sel | gpt-4o-mini-2024-07-18       |       0 |  20 | 0.864 ±0.087  | 0.850 ±0.161          |
| probe_random_sel | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.864 ±0.075  | 0.700 ±0.206          |
| probe_random_sel | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.834 ±0.056  | 0.550 ±0.224          |
| probe_random_sel | gpt-4o-mini-2024-07-18       |       3 |  20 | 0.871 ±0.043  | 0.350 ±0.214          |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |       0 |  20 | 0.807 ±0.088  | 0.750 ±0.195          |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.893 ±0.063  | 0.900 ±0.135          |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.851 ±0.066  | 0.600 ±0.220          |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |       3 |  20 | 0.856 ±0.050  | 0.350 ±0.214          |
| probe_skel_only  | none                         |       0 |  20 | 0.766 ±0.114  | 0.350 ±0.214          |
| probe_skel_only  | none                         |       1 |  20 | 0.956 ±0.032  | 0.700 ±0.206          |
| probe_skel_only  | none                         |       2 |  20 | 0.901 ±0.065  | 0.550 ±0.224          |
| probe_skel_only  | none                         |       3 |  20 | 0.872 ±0.063  | 0.350 ±0.214          |

## Table 5 — quality per token

| arm              | model_tag                    |   n | directed_f1   | total_tokens      | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:-----------------|:-----------------------------|----:|:--------------|:------------------|:-------------|:-------------|-------------------:|
| probe            | gpt-4o-mini-2024-07-18       |  80 | 0.913 ±0.027  | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 |             0.8222 |
| probe            | qwen3-coder-30b-a3b-instruct |  80 | 0.922 ±0.028  | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4992 |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |  80 | 0.919 ±0.027  | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 |             0.8276 |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.914 ±0.031  | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4948 |
| probe_random_sel | gpt-4o-mini-2024-07-18       |  80 | 0.858 ±0.033  | 1110.888 ±74.502  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7724 |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |  80 | 0.852 ±0.034  | 1847.400 ±133.332 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4611 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm              | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-----------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe            | gpt-4o-mini-2024-07-18       |  80 | 3.388 ±0.193    | 2.150 ±0.286 | 0.913 ±0.027  | 0.960 ±0.014            |
| probe            | qwen3-coder-30b-a3b-instruct |  80 | 2.625 ±0.328    | 2.987 ±0.295 | 0.922 ±0.028  | 0.964 ±0.013            |
| probe_maxdeg_sel | gpt-4o-mini-2024-07-18       |  80 | 3.388 ±0.193    | 2.150 ±0.286 | 0.919 ±0.027  | 0.960 ±0.014            |
| probe_maxdeg_sel | qwen3-coder-30b-a3b-instruct |  80 | 2.625 ±0.328    | 2.987 ±0.295 | 0.914 ±0.031  | 0.964 ±0.013            |
| probe_random_sel | gpt-4o-mini-2024-07-18       |  80 | 3.388 ±0.193    | 2.150 ±0.286 | 0.858 ±0.033  | 0.960 ±0.014            |
| probe_random_sel | qwen3-coder-30b-a3b-instruct |  80 | 2.625 ±0.328    | 2.987 ±0.295 | 0.852 ±0.034  | 0.964 ±0.013            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |   probe_random_sel |
|-------:|--------:|-------------------:|
|      1 |   1.051 |              1.481 |
|      2 |   0.458 |              1.007 |
|      3 |   0.236 |              0.648 |
