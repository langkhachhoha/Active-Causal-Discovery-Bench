# Study 1 — selection vs inference

240 successful episodes from `traces/study1/ablation_tightbudget`.

## Table 1 — every arm (mean ± 95% CI over paired instances)

| arm         | model_tag                    |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | interventions_used   | efficiency   | prompt_tokens     | completion_tokens   | total_tokens      | cost_usd     | llm_calls    | wall_sec     |
|:------------|:-----------------------------|----:|:--------------|:---------------|:--------------|:-------------|:---------------------|:-------------|:------------------|:--------------------|:------------------|:-------------|:-------------|:-------------|
| eig+meek    | none                         |  40 | 0.838 ±0.059  | 0.925 ±0.049   | 0.938 ±0.020  | 1.325 ±0.361 | 1.600 ±0.208         | 1.000 ±0.000 | —                 | —                   | —                 | —            | —            | 0.162 ±0.286 |
| llm+meek    | gpt-4o-mini-2024-07-18       |  40 | 0.831 ±0.060  | 0.925 ±0.049   | 0.938 ±0.020  | 1.400 ±0.363 | 1.625 ±0.218         | 1.000 ±0.000 | 831.425 ±136.721  | 69.400 ±9.360       | 900.825 ±145.772  | 0.000 ±0.000 | 1.625 ±0.218 | 2.330 ±0.822 |
| llm+meek    | qwen3-coder-30b-a3b-instruct |  40 | 0.841 ±0.058  | 0.915 ±0.062   | 0.938 ±0.020  | 1.325 ±0.380 | 1.600 ±0.208         | 1.000 ±0.000 | 1245.400 ±189.788 | 86.600 ±11.011      | 1332.000 ±200.299 | 0.000 ±0.000 | 1.600 ±0.208 | 4.256 ±0.793 |
| maxdeg+meek | none                         |  40 | 0.837 ±0.059  | 0.925 ±0.049   | 0.938 ±0.020  | 1.325 ±0.367 | 1.600 ±0.208         | 1.000 ±0.000 | —                 | —                   | —                 | —            | —            | 0.163 ±0.286 |
| oracle+meek | none                         |  40 | 0.833 ±0.060  | 0.925 ±0.049   | 0.938 ±0.020  | 1.275 ±0.384 | 1.575 ±0.209         | 1.000 ±0.000 | —                 | —                   | —                 | —            | —            | 0.164 ±0.286 |
| random+meek | none                         |  40 | 0.809 ±0.055  | 0.915 ±0.062   | 0.938 ±0.020  | 1.725 ±0.391 | 1.625 ±0.218         | 1.000 ±0.000 | —                 | —                   | —                 | —            | —            | 0.165 ±0.286 |

## Table 2 — SELECTION quality: inference held fixed at `meek`

| selector   | model_tag                    |   n | directed_f1   | selection_regret_total   | selection_regret_mean   | selection_quality_mean   | eig_regret_total   | wasted_steps   | steps_taken   |
|:-----------|:-----------------------------|----:|:--------------|:-------------------------|:------------------------|:-------------------------|:-------------------|:---------------|:--------------|
| eig        | none                         |  40 | 0.838 ±0.059  | 0.175 ±0.119             | 0.158 ±0.112            | 0.952 ±0.034             | 0.000 ±0.000       | 0.000 ±0.000   | 1.600 ±0.208  |
| llm        | gpt-4o-mini-2024-07-18       |  40 | 0.831 ±0.060  | 0.475 ±0.272             | 0.298 ±0.161            | 0.907 ±0.048             | 0.126 ±0.071       | 0.000 ±0.000   | 1.625 ±0.218  |
| llm        | qwen3-coder-30b-a3b-instruct |  40 | 0.841 ±0.058  | 0.175 ±0.138             | 0.154 ±0.133            | 0.953 ±0.038             | 0.016 ±0.020       | 0.000 ±0.000   | 1.600 ±0.208  |
| maxdeg     | none                         |  40 | 0.837 ±0.059  | 0.150 ±0.132             | 0.145 ±0.133            | 0.956 ±0.038             | 0.007 ±0.008       | 0.000 ±0.000   | 1.600 ±0.208  |
| oracle     | none                         |  40 | 0.833 ±0.060  | 0.000 ±0.000             | 0.000 ±0.000            | 1.000 ±0.000             | 0.139 ±0.078       | 0.000 ±0.000   | 1.575 ±0.209  |
| random     | none                         |  40 | 0.809 ±0.055  | 1.200 ±0.609             | 0.667 ±0.302            | 0.798 ±0.078             | 0.354 ±0.110       | 0.000 ±0.000   | 1.625 ±0.218  |

## Table 3 — INFERENCE quality: selection held fixed at `oracle`

| inferencer   | model_tag   |   n | directed_f1   | compelled_f1   | skeleton_f1   | dag_shd      | orientation_accuracy   | submit_directed   | submit_undirected   |
|:-------------|:------------|----:|:--------------|:---------------|:--------------|:-------------|:-----------------------|:------------------|:--------------------|
| meek         | none        |  40 | 0.833 ±0.060  | 0.925 ±0.049   | 0.938 ±0.020  | 1.275 ±0.384 | 0.910 ±0.066           | 6.350 ±0.704      | 0.025 ±0.049        |

## Table 4 — the full selector x inferencer grid (directed F1)

| ('selector', '')   |   ('meek', 'gpt-4o-mini-2024-07-18') |   ('meek', 'none') |   ('meek', 'qwen3-coder-30b-a3b-instruct') |
|:-------------------|-------------------------------------:|-------------------:|-------------------------------------------:|
| eig                |                              nan     |              0.838 |                                    nan     |
| llm                |                                0.831 |            nan     |                                      0.841 |
| maxdeg             |                              nan     |              0.837 |                                    nan     |
| oracle             |                              nan     |              0.833 |                                    nan     |
| random             |                              nan     |              0.809 |                                    nan     |

## Table 5 — attributing the end-to-end gap

| model                        |   best_possible (oracle+meek) |   full_llm_agent (llm+llm) |   total_gap |   selection_gap (oracle+meek - llm+meek) |   inference_gap (oracle+meek - oracle+llm) |   llm_e2e (no scaffold) |
|:-----------------------------|------------------------------:|---------------------------:|------------:|-----------------------------------------:|-------------------------------------------:|------------------------:|
| gpt-4o-mini-2024-07-18       |                        0.8327 |                        nan |         nan |                                   0.002  |                                        nan |                     nan |
| qwen3-coder-30b-a3b-instruct |                        0.8327 |                        nan |         nan |                                  -0.0083 |                                        nan |                     nan |

Read Table 5 as: `selection_gap` is what you lose by letting the LLM pick experiments (inference held perfect); `inference_gap` is what you lose by letting the LLM read the results (selection held perfect). The larger term is the bottleneck.

## Table 6 — scaling with graph size

| arm         | model_tag                    |   level |   n | directed_f1   | selection_regret_total   | cost_usd     |
|:------------|:-----------------------------|--------:|----:|:--------------|:-------------------------|:-------------|
| eig+meek    | none                         |       0 |  10 | 0.743 ±0.166  | 0.300 ±0.299             | —            |
| eig+meek    | none                         |       1 |  10 | 0.793 ±0.127  | 0.100 ±0.196             | —            |
| eig+meek    | none                         |       2 |  10 | 0.900 ±0.065  | 0.100 ±0.196             | —            |
| eig+meek    | none                         |       3 |  10 | 0.915 ±0.049  | 0.200 ±0.261             | —            |
| llm+meek    | gpt-4o-mini-2024-07-18       |       0 |  10 | 0.743 ±0.166  | 0.300 ±0.299             | 0.000 ±0.000 |
| llm+meek    | gpt-4o-mini-2024-07-18       |       1 |  10 | 0.772 ±0.137  | 0.400 ±0.433             | 0.000 ±0.000 |
| llm+meek    | gpt-4o-mini-2024-07-18       |       2 |  10 | 0.899 ±0.068  | 0.200 ±0.392             | 0.000 ±0.000 |
| llm+meek    | gpt-4o-mini-2024-07-18       |       3 |  10 | 0.909 ±0.046  | 1.000 ±0.826             | 0.000 ±0.000 |
| llm+meek    | qwen3-coder-30b-a3b-instruct |       0 |  10 | 0.743 ±0.166  | 0.300 ±0.299             | 0.000 ±0.000 |
| llm+meek    | qwen3-coder-30b-a3b-instruct |       1 |  10 | 0.819 ±0.128  | 0.000 ±0.000             | 0.000 ±0.000 |
| llm+meek    | qwen3-coder-30b-a3b-instruct |       2 |  10 | 0.907 ±0.068  | 0.000 ±0.000             | 0.000 ±0.000 |
| llm+meek    | qwen3-coder-30b-a3b-instruct |       3 |  10 | 0.896 ±0.054  | 0.400 ±0.433             | 0.001 ±0.000 |
| maxdeg+meek | none                         |       0 |  10 | 0.743 ±0.166  | 0.300 ±0.299             | —            |
| maxdeg+meek | none                         |       1 |  10 | 0.782 ±0.124  | 0.200 ±0.392             | —            |
| maxdeg+meek | none                         |       2 |  10 | 0.907 ±0.068  | 0.000 ±0.000             | —            |
| maxdeg+meek | none                         |       3 |  10 | 0.915 ±0.049  | 0.100 ±0.196             | —            |
| oracle+meek | none                         |       0 |  10 | 0.700 ±0.153  | 0.000 ±0.000             | —            |
| oracle+meek | none                         |       1 |  10 | 0.802 ±0.131  | 0.000 ±0.000             | —            |
| oracle+meek | none                         |       2 |  10 | 0.919 ±0.070  | 0.000 ±0.000             | —            |
| oracle+meek | none                         |       3 |  10 | 0.909 ±0.062  | 0.000 ±0.000             | —            |
| random+meek | none                         |       0 |  10 | 0.719 ±0.162  | 0.300 ±0.299             | —            |
| random+meek | none                         |       1 |  10 | 0.795 ±0.109  | 0.200 ±0.261             | —            |
| random+meek | none                         |       2 |  10 | 0.866 ±0.069  | 1.100 ±0.898             | —            |
| random+meek | none                         |       3 |  10 | 0.855 ±0.060  | 3.200 ±1.724             | —            |

## Table 7 — mean selection regret per intervention index

|   step |   eig |   llm |   maxdeg |   oracle |   random |
|-------:|------:|------:|---------:|---------:|---------:|
|      1 | 0.184 | 0.329 |    0.158 |        0 |    0.921 |
|      2 | 0     | 0.021 |    0     |        0 |    0.458 |
|      3 | 0     | 0     |    0     |        0 |    0.667 |
