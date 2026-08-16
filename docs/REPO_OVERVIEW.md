# ACDB — Repo Overview (đọc file này thay vì đọc lại code)

*Cập nhật: 2026-08-16. Mô tả trạng thái repo **trước khi** thêm hai study mới (`docs/IDEAS.md`).*

---

## 1. Repo này làm gì

**ACDB (Active Causal Discovery Benchmark)** đo xem một LLM agent có thể **khám phá cấu trúc nhân quả**
hay không, dưới ràng buộc ngân sách thí nghiệm.

Mỗi instance là một **linear-Gaussian SCM ẩn** trên `d` biến:

```
X_i = Σ_{j ∈ Pa(i)} w_ij X_j + ε_i ,    ε_i ~ N(0, σ_i²)
```

Agent chỉ nhìn thấy: tên biến ẩn danh (`X0..X{d-1}`), **một** ma trận dữ liệu quan sát, và dữ liệu can thiệp
mà nó tự xin. Evaluator giữ toàn bộ sự thật: DAG `G`, CPDAG(`G`), tham số SCM, và tập can thiệp tối thiểu `I*`.

**Protocol cố định:** `observe()` (đúng 1 lần) → `intervene(var, value)` (mỗi lần tốn 1 budget) → `submit_graph()` (kết thúc).

**Điểm mấu chốt về mặt lý thuyết:** dữ liệu quan sát chỉ định danh được **Markov equivalence class** (CPDAG),
không phải một DAG duy nhất. Muốn ra DAG thật thì **bắt buộc phải can thiệp**. Đó là lý do benchmark
chấm điểm theo nhiều lớp.

---

## 2. Bốn lớp chấm điểm

| Metric | So với | Ý nghĩa |
|---|---|---|
| `skeleton_f1` | skeleton của CPDAG(G) | phục hồi cạnh (bỏ qua hướng) |
| `compelled_f1` | cạnh **có hướng** của CPDAG(G) | phần hướng mà dữ liệu quan sát *được phép* xác định |
| `directed_f1` | cạnh của DAG `G` | phục hồi hướng đầy đủ |
| `dag_shd` | DAG `G` | khoảng cách sửa cạnh (thiếu/thừa/ngược/chưa định hướng đều = 1 lỗi) |
| `efficiency` | `|I*|` | `|I*| / max(dùng, |I*|)` — tiết kiệm ngân sách |

Nộp cạnh **vô hướng** là hợp lệ và khác với **bỏ sót** cạnh (nó ăn điểm skeleton, mất điểm directed).

**Random floor** (closed-form, `M = d(d-1)/2`): `E[F1] = (1/(M+1)) Σ_{m=0..M} k·m / (M·(m+k))`.
Nằm quanh 0.12–0.22 tuỳ level. Số nào không vượt floor thì vô nghĩa.

---

## 3. Cách một instance được sinh ra (`benchmark/instance.py`)

```
sample_random_dag(d, k)                     # k cạnh forward theo một topo order ngẫu nhiên
  → dag_to_cpdag()                          # collider + Meek closure (rule 1-3)
  → reject_graph()                          # loại nếu CPDAG có <2 cạnh vô hướng,
                                            #   hoặc mọi cạnh vô hướng chung 1 node, hoặc skeleton rời rạc
  → parameterize_linear_gaussian_scm()      # w ~ U(-2,-0.5) ∪ U(0.5,2), σ² = noise_var
  → reject_scm()                            # loại nếu covariance gần suy biến, hoặc vi phạm
                                            #   faithfulness (|partial corr| < eps = 0.1)
  → compute_minimum_intervention_set()      # brute-force: tập nhỏ nhất mà can thiệp vào đó
                                            #   + Meek closure (rule 1-4) là ra đúng G
  → reject_intervention_profile()           # loại nếu |I*| == 0
  → random permutation nhãn node            # chống leak thứ tự topo
  → sample n_obs rows
budget = |I*| + budget_slack
```

Toàn bộ deterministic theo `seed`. Cùng seed ⇒ cùng instance ⇒ **kết quả các method là paired**.

---

## 4. Các method đã có trong `run_ladder.py`

| Method | Panel | Mô tả |
|---|---|---|
| `pc` | observational | PC (causal-learn, Fisher-Z, α=0.05) chỉ trên dữ liệu quan sát → nộp CPDAG |
| `pc_greedy` | active | PC → còn cạnh vô hướng thì can thiệp **node có bậc vô hướng cao nhất**, định hướng bằng mean-shift test |
| `llm_raw` | active | LLM nhận ma trận dữ liệu thô, được `intervene` + `submit_graph` |
| `llm_stats` | active | như trên nhưng thêm `correlation` / `partial_correlation` / `independence_test` |
| `pc_cpdag_llm` | active | đưa sẵn đồ thị PC cho LLM, LLM chỉ can thiệp để gỡ hướng |
| `llm_stats_cpdag_greedy` | active | LLM (chỉ stats, không can thiệp) → rồi greedy can thiệp gỡ nốt |
| `oracle` | active | nộp thẳng `G` → trần 1.000 |

Baseline `pc_greedy` dùng quy tắc định hướng `z_calibrated_mean_shift`:
can thiệp `do(a = μ_a + 3.0)`, biến `b` được coi là bị dịch nếu
`|μ_b^int − μ_b^obs| > 1.96·sqrt(σ²_b,obs/n_obs + σ²_b,int/n_int)`; nếu dịch thì `a→b`, nếu không thì `b→a`
(có guard chống tạo chu trình). **Không** áp Meek closure sau khi định hướng.

---

## 5. Kết quả headline đã có (directed-F1, trung bình 6 level, d = 4…14)

| Model | `llm_raw` | `llm_stats` | `+cpdag_greedy` | `pc_greedy` | `oracle` |
|---|:-:|:-:|:-:|:-:|:-:|
| GPT-5.5 | **0.748** | 0.700 | 0.474 | 0.709 | 1.000 |
| Sonnet 4.6 | 0.346 | 0.213 | 0.350 | 0.709 | 1.000 |
| Gemini 3 Flash | 0.323 | 0.279 | 0.510 | 0.709 | 1.000 |
| GPT-5.4-mini | 0.158 | 0.142 | 0.125 | 0.709 | 1.000 |
| Haiku 4.5 | 0.149 | 0.166 | 0.313 | 0.709 | 1.000 |

Chỉ GPT-5.5 raw vượt được baseline cổ điển. Đây là thông điệp chính: **chưa giải quyết được**.

---

## 6. Bản đồ code

```
run_ladder.py                        runner chính, monolith 1.3k dòng: level ladder,
                                     preflight seed map, checkpoint/resume, CSV + events.jsonl
run_random_dag_baseline.py           random uniform-m floor (Appendix C)

src/causal_discovery/
  config/v1.py          BenchmarkConfig (d, k, n_obs, n_int, weight_range, noise_var,
                        faithfulness_eps, budget_slack); mặc định k = d+1, n_int = n_obs
  core/dag.py           DAG bất biến (frozenset edges), acyclic check, relabel
  core/scm.py           LinearGaussianSCM (weights, noise_variances, intercepts)
  core/permutation.py   hoán vị nhãn node
  graph_gen/random_dag.py   sample_random_dag(d, k)
  scm/generation.py     parameterize_linear_gaussian_scm
  scm/diagnostics.py    implied_covariance, is_near_singular, partial correlations
  equivalence/cpdag.py  CPDAG bất biến + canonical_undirected_edge(a,b) = (min,max)
  equivalence/theory.py **trái tim lý thuyết**: dag_to_cpdag, Meek rules 1-4,
                        compute_minimum_intervention_set, các reject_* policy
  sampling/sampler.py   sample_observational_data / sample_interventional_data (hard do())
  benchmark/instance.py lắp ráp instance (mục 3)
  runtime/session.py    BenchmarkEnv: observe 1 lần, intervene trừ budget, submit seal
  scoring/submission.py GraphSubmission (validate: không self-loop, không 2-cycle,
                        không vừa directed vừa undirected, phải acyclic)
  scoring/scores.py     4 lớp metric ở mục 2
  agents/prompts.py     system/session prompt cho raw & stats agent
  agents/tool_schema.py JSON schema strict cho 1 tool `causal_discovery_action`
  agents/litellm_model.py  adapter LiteLLM, ép tool_choice, đếm token/cost/latency
  agents/llm.py         LLMRawAgent / LLMStatsAgent + parse action
  agents/session.py     vòng lặp session không-LLM-specific
  baselines/cpdag_parser.py  đọc endpoint matrix của causal-learn

scripts/extract_trace_rows.py          events.jsonl → CSV per-step
scripts/ladder_random_floor_sanity.py  Monte-Carlo calibrate random floor
traces/ladder/full_*                   5 panel model canonical
traces/aggregated/                     bảng tổng hợp per model × level × method
```

---

## 7. Những chi tiết dễ quên nhưng quan trọng

- **`.env` hiện lưu key OpenRouter dưới tên `OPENAI_API_KEY`** (`sk-or-v1-…`). Code mới trong
  `src/causal_discovery/active/` chấp nhận cả `OPENROUTER_API_KEY` lẫn `OPENAI_API_KEY`.
- `run_ladder.py` mặc định `--env-file` trỏ **ra ngoài repo** (`parent.parent/.env`) — luôn truyền `--env-file .env`.
- `runtime_seed = seed*10000 + level*101 + 7`; instance được cache theo `(level, seed)` nên mọi method
  trong cùng một run dùng **đúng cùng một world**.
- Budget của `pc_greedy` không được tính là "LLM cost" (token = 0), nhưng `interventions_used` vẫn tính.
- `_sanitize_submission_edges` bỏ cạnh vô hướng trùng với cạnh có hướng và log `sanitized_overlap_count`.
- Ở bước cuối (`step == max_steps`) agent bị **ép** chỉ được `submit_graph`.
- `efficiency` phạt dùng **quá** `|I*|`, không thưởng dùng ít hơn (dùng ít hơn thì thường F1 tụt).
- Meek rule 4 **chỉ** được bật khi tính `I*` (bối cảnh có can thiệp), không bật trong `dag_to_cpdag`.

---

## 8. Điểm yếu / chỗ còn trống (chi tiết ở `docs/IDEAS.md`)

1. Chỉ có **một con số end-to-end**. Không tách được "chọn thí nghiệm dở" khỏi "suy luận dở".
2. Baseline chọn thí nghiệm chỉ là heuristic bậc cao nhất — **chưa có baseline experimental design
   đúng nghĩa** (information gain trên Markov equivalence class).
3. Agent **không có belief state**. Nó không giữ phân phối trên đồ thị, không cập nhật Bayes.
4. Không đo **chất lượng của từng quyết định thí nghiệm** (chỉ đo kết quả cuối).
5. `pc_greedy` không áp Meek closure sau mỗi can thiệp ⇒ baseline cổ điển đang bị **thiệt**.
