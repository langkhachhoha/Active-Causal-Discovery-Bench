# Study 2 — PROBE

280 successful episodes from `study2/robust_alpha0.10`.

## Table 1 — main results (mean ± 95% CI over paired instances)

| arm                | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | efficiency   | interventions_used   | prompt_tokens    | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec     |
|:-------------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:-------------|:---------------------|:-----------------|:--------------------|:------------------|:-------------|:-------------|:-------------|
| oracle             | none                         |  40 | 1.000 ±0.000  | 1.000 ±0.000   | 1.000 ±0.000  | 0.000 ±0.000 | 1.000 ±0.000 | 0.000 ±0.000         | —                | —                   | —                 | —            | —            | 0.005 ±0.002 |
| pc_greedy_meek     | none                         |  40 | 0.764 ±0.056  | 0.761 ±0.090   | 0.885 ±0.023  | 2.275 ±0.427 | 0.883 ±0.064 | 1.625 ±0.218         | —                | —                   | —                 | —            | —            | 0.212 ±0.260 |
| probe              | gpt-4o-mini-2024-07-18       |  40 | 0.857 ±0.037  | 0.863 ±0.059   | 0.894 ±0.022  | 1.700 ±0.416 | 0.725 ±0.064 | 2.275 ±0.198         | 979.825 ±46.933  | 103.675 ±5.345      | 1083.500 ±45.831  | 0.000 ±0.000 | 1.000 ±0.000 | 2.888 ±0.282 |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 0.843 ±0.050  | 0.860 ±0.064   | 0.894 ±0.026  | 1.750 ±0.485 | 0.738 ±0.067 | 2.275 ±0.157         | 1461.550 ±63.356 | 333.300 ±81.466     | 1794.850 ±109.716 | 0.001 ±0.000 | 1.000 ±0.000 | 4.411 ±0.765 |
| probe_oracle_edits | none                         |  40 | 0.983 ±0.019  | 0.984 ±0.023   | 0.990 ±0.009  | 0.175 ±0.184 | 0.867 ±0.065 | 1.950 ±0.156         | —                | —                   | —                 | —            | —            | 0.653 ±0.272 |
| probe_random_edits | none                         |  40 | 0.795 ±0.057  | 0.820 ±0.072   | 0.875 ±0.027  | 2.150 ±0.514 | 0.717 ±0.062 | 2.300 ±0.201         | —                | —                   | —                 | —            | —            | 0.859 ±0.276 |
| probe_skel_only    | none                         |  40 | 0.839 ±0.039  | 0.832 ±0.065   | 0.885 ±0.023  | 1.775 ±0.368 | 0.833 ±0.069 | 1.900 ±0.183         | —                | —                   | —                 | —            | —            | 0.797 ±0.286 |

## Table 2 — hypothesis-space quality drives everything

| hypothesis_source     | model_tag                    |   n | directed_f1   | truth_in_hypotheses   | best_f1_in_hypotheses   | n_hypotheses   | truth_rank_final   |
|:----------------------|:-----------------------------|----:|:--------------|:----------------------|:------------------------|:---------------|:-------------------|
| llm_repair + pc_mec   | gpt-4o-mini-2024-07-18       |  40 | 0.857 ±0.037  | 0.200 ±0.126          | 0.898 ±0.026            | 48.000 ±0.000  | -0.575 ±0.271      |
| llm_repair + pc_mec   | qwen3-coder-30b-a3b-instruct |  40 | 0.843 ±0.050  | 0.250 ±0.136          | 0.898 ±0.029            | 48.000 ±0.000  | -0.450 ±0.313      |
| oracle edits (no LLM) | none                         |  40 | 0.983 ±0.019  | 0.975 ±0.049          | 0.998 ±0.004            | 48.000 ±0.000  | 1.050 ±0.156       |
| pc_skeleton (no LLM)  | none                         |  40 | 0.839 ±0.039  | 0.125 ±0.104          | 0.878 ±0.027            | 41.000 ±3.469  | -0.750 ±0.208      |
| random edits (no LLM) | none                         |  40 | 0.795 ±0.057  | 0.175 ±0.119          | 0.883 ±0.027            | 48.000 ±0.000  | -0.625 ±0.260      |

`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; `best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together they cap what the decision layer can possibly deliver.

## Table 3 — decision-layer ablations (hypothesis space held fixed)

| arm   | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | map_weight_final   | entropy_final_nats   |
|:------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------------|:---------------------|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.857 ±0.037  | 0.863 ±0.059   | 0.894 ±0.022  | 1.700 ±0.416 | 2.275 ±0.198         | 0.942 ±0.039       | 0.134 ±0.079         |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.843 ±0.050  | 0.860 ±0.064   | 0.894 ±0.026  | 1.750 ±0.485 | 2.275 ±0.157         | 0.908 ±0.057       | 0.221 ±0.123         |

## Table 4 — scaling with graph size

| arm                | model_tag                    |   level |   n | directed_f1   | truth_in_hypotheses   |
|:-------------------|:-----------------------------|--------:|----:|:--------------|:----------------------|
| oracle             | none                         |       1 |  20 | 1.000 ±0.000  | —                     |
| oracle             | none                         |       2 |  20 | 1.000 ±0.000  | —                     |
| pc_greedy_meek     | none                         |       1 |  20 | 0.738 ±0.090  | —                     |
| pc_greedy_meek     | none                         |       2 |  20 | 0.790 ±0.067  | —                     |
| probe              | gpt-4o-mini-2024-07-18       |       1 |  20 | 0.886 ±0.044  | 0.300 ±0.206          |
| probe              | gpt-4o-mini-2024-07-18       |       2 |  20 | 0.827 ±0.058  | 0.100 ±0.135          |
| probe              | qwen3-coder-30b-a3b-instruct |       1 |  20 | 0.845 ±0.079  | 0.400 ±0.220          |
| probe              | qwen3-coder-30b-a3b-instruct |       2 |  20 | 0.841 ±0.063  | 0.100 ±0.135          |
| probe_oracle_edits | none                         |       1 |  20 | 0.978 ±0.028  | 0.950 ±0.098          |
| probe_oracle_edits | none                         |       2 |  20 | 0.988 ±0.024  | 1.000 ±0.000          |
| probe_random_edits | none                         |       1 |  20 | 0.765 ±0.092  | 0.200 ±0.180          |
| probe_random_edits | none                         |       2 |  20 | 0.825 ±0.066  | 0.150 ±0.161          |
| probe_skel_only    | none                         |       1 |  20 | 0.847 ±0.058  | 0.200 ±0.180          |
| probe_skel_only    | none                         |       2 |  20 | 0.832 ±0.053  | 0.050 ±0.098          |

## Table 5 — quality per token

| arm   | model_tag                    |   n | directed_f1   | total_tokens      | cost_usd     | llm_calls    |   f1_per_1k_tokens |
|:------|:-----------------------------|----:|:--------------|:------------------|:-------------|:-------------|-------------------:|
| probe | gpt-4o-mini-2024-07-18       |  40 | 0.857 ±0.037  | 1083.500 ±45.831  | 0.000 ±0.000 | 1.000 ±0.000 |             0.7907 |
| probe | qwen3-coder-30b-a3b-instruct |  40 | 0.843 ±0.050  | 1794.850 ±109.716 | 0.001 ±0.000 | 1.000 ±0.000 |             0.4697 |

## Table 6 — how aggressively each model edits PC's skeleton

| arm                | model_tag                    |   n | repair_remove   | repair_add   | directed_f1   | best_f1_in_hypotheses   |
|:-------------------|:-----------------------------|----:|:----------------|:-------------|:--------------|:------------------------|
| probe              | gpt-4o-mini-2024-07-18       |  40 | 3.725 ±0.157    | 2.825 ±0.371 | 0.857 ±0.037  | 0.898 ±0.026            |
| probe              | qwen3-coder-30b-a3b-instruct |  40 | 3.200 ±0.331    | 3.400 ±0.231 | 0.843 ±0.050  | 0.898 ±0.029            |
| probe_oracle_edits | none                         |  40 | 0.275 ±0.140    | 1.200 ±0.224 | 0.983 ±0.019  | 0.998 ±0.004            |
| probe_random_edits | none                         |  40 | 3.975 ±0.049    | 4.000 ±0.000 | 0.795 ±0.057  | 0.883 ±0.027            |

## Table 7 — posterior entropy after each experiment

|   step |   probe |
|-------:|--------:|
|      1 |   0.993 |
|      2 |   0.318 |
|      3 |   0.178 |
