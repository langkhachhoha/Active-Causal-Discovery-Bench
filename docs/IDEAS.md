# Hai ý tưởng workshop phát triển từ ACDB

*Cả hai đều theo hướng **active experiment**: LLM không chỉ đề xuất lời giải mà còn phải chọn thí nghiệm tiếp theo.*

---

## Bối cảnh: 5 khoảng trống trong repo gốc

| # | Khoảng trống | Ý tưởng khai thác |
|---|---|---|
| 1 | Chỉ có **một con số end-to-end** (`directed_f1`). Không biết agent dở ở khâu *chọn thí nghiệm* hay khâu *suy luận từ kết quả*. | Study 1 |
| 2 | Baseline chọn thí nghiệm chỉ là heuristic "bậc vô hướng cao nhất" — **chưa có experimental design đúng nghĩa** (information gain trên Markov equivalence class). | Study 1 + 2 |
| 3 | Agent **không có belief state**: không giữ phân phối trên đồ thị, không cập nhật Bayes sau mỗi can thiệp. | Study 2 |
| 4 | Không đo **chất lượng từng quyết định thí nghiệm**, chỉ đo kết quả cuối. | Study 1 |
| 5 | `pc_greedy` không áp Meek closure sau can thiệp ⇒ baseline cổ điển đang **bị thiệt**. | sửa trong cả hai |

---

# Study 1 — *Tách bạch: chọn thí nghiệm hay suy luận, cái nào hỏng?*

**File:** `run_study1_decompose.py` · **Chạy:** `bash scripts/study1.sh main`

### Câu chuyện

Một agent active causal discovery làm **hai việc khác nhau**: (a) *chọn thí nghiệm nào để chạy*,
và (b) *suy ra cấu trúc từ kết quả*. Một điểm số end-to-end không thể nói cái nào hỏng.
Bài này phân tách agent thành lưới **selector × inferencer**, chạy toàn bộ tích Descartes trên
cùng một tập instance, rồi **quy trách nhiệm** cho từng khâu.

### Thiết kế

```
selector    ∈ { random, maxdeg, eig, llm, oracle }
inferencer  ∈ { meek, llm }                          → lưới 5 × 2 = 10 arm
                                                     + llm_e2e (agent nguyên bản, không scaffold)
```

Mọi arm **dùng chung**: cùng instance, cùng front-end PC, cùng giá trị can thiệp
(`mean + 3·sd`), cùng cơ chế cập nhật belief. Nên chênh lệch giữa các arm là **quy được**.

| Thành phần | Ý nghĩa |
|---|---|
| `random` | sàn — chọn ngẫu nhiên node còn cạnh vô hướng |
| `maxdeg` | heuristic của repo gốc |
| `eig` | **Bayesian OED**: liệt kê chính xác MEC hiện tại, chọn target tối đa hoá entropy của phân hoạch |
| `llm` | LLM nhìn PDAG hiện tại + tóm tắt bằng chứng, chọn target |
| `oracle` | trần — biết `G`, chọn target gỡ được nhiều cạnh nhất |
| `meek` | mean-shift test + **Meek closure** (sửa khoảng trống #5) |
| `llm` (infer) | LLM nhận PDAG ban đầu + thống kê đủ của mọi can thiệp, tự nộp đồ thị |

### Metric mới (đây là đóng góp đo lường)

- **`selection_regret`** — tại mỗi bước, vì evaluator giữ `G`, ta tính chính xác
  `gain(a)` = số cạnh vô hướng được gỡ nếu can thiệp vào `a` (sau Meek closure).
  `regret = max_a gain(a) − gain(a_chọn)`. Đây là chấm điểm **từng quyết định**, không phải kết quả cuối.
- **`eig_regret`** — cùng ý tưởng nhưng theo nats thông tin, không cần biết `G`.
- **`orientation_accuracy`** — trong các cạnh vừa được định hướng, bao nhiêu % đúng.
- **`pc_skeleton_f1_ceiling`** — trần chung của mọi arm, để biết còn bao nhiêu chỗ để cải thiện.
- **`mec_size` / `mec_entropy`** — độ khó thật sự của instance.

### Bảng chốt bài (Table 5 trong `analysis/tables.md`)

```
total_gap     = (oracle + meek)  −  (llm + llm)
selection_gap = (oracle + meek)  −  (llm + meek)      ← mất mát do CHỌN dở
inference_gap = (oracle + meek)  −  (oracle + llm)    ← mất mát do SUY LUẬN dở
```

### Kết quả sơ bộ (smoke, d = 4 & 6, 2 seed/level — đã chạy thật)

| model | trần (oracle+meek) | agent đầy đủ (llm+llm) | tổng gap | **selection gap** | **inference gap** | llm_e2e |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| gpt-4o-mini | 0.964 | 0.401 | 0.563 | **0.000** | **0.519** | 0.000 |
| qwen3-coder-30b | 0.964 | 0.684 | 0.280 | **0.000** | **0.267** | 0.188 |

> **Thông điệp:** LLM nhỏ **chọn thí nghiệm tốt ngang oracle** (regret = 0), nhưng
> **không đọc nổi kết quả**. Toàn bộ khoảng cách nằm ở khâu suy luận.
> Và `llm_e2e` gần 0 cho thấy riêng phần scaffold đã đáng giá rất nhiều.

### Ablation/analysis có sẵn

- regret theo từng bước (`fig_regret.png`, Table 7)
- scaling theo `d` = 4, 6, 8, 10, 12
- `n_obs` ∈ {60, 300, 1000} — chẩn đoán có đổi khi dữ liệu khan/dồi dào không
- `--evidence-mode raw|summary` — LLM đọc số thô có khá hơn đọc thống kê đủ không
- token in/out, cached tokens, cost USD, số call, latency, tỉ lệ phải repair JSON
- `select_wasted_targets` (chọn node không còn cạnh vô hướng), `select_repeat_targets`

---

# Study 2 — *NemChua: LLM đề xuất không gian giả thuyết, quyết định để cho toán học*

**File:** `run_study2_probe.py` · **Chạy:** `bash scripts/study2.sh main`

### Câu chuyện

Study 1 nói: LLM dở ở **suy luận số**, không dở ở **chọn thí nghiệm**. Vậy thiết kế lại vai trò
của nó. NemChua tách agent làm 3 phần, LLM chỉ giữ đúng phần nó giỏi:

1. **Propose** — LLM **không** bịa đồ thị từ đầu. Nó **kiểm toán skeleton của PC**:
   nhìn ma trận tương quan + tương quan riêng phần (điều kiện trên mọi biến còn lại) và chỉ ra
   cạnh nào PC có thể đã thêm nhầm / bỏ sót. Đây là việc *cục bộ, có cấu trúc* — vừa sức model nhỏ.
2. **Score** — mỗi ứng viên được chấm bằng **BIC linear-Gaussian chính xác** trên dữ liệu quan sát
   ⇒ posterior `w` trên tập giả thuyết. Đề xuất tồi bị dìm **bằng cơ học**, không cần tin LLM.
3. **Choose & Update** — can thiệp tiếp theo tối đa hoá **expected information gain**
   `I(H ; Y | do(a))`; kết quả cập nhật `w` bằng Bayes với **likelihood can thiệp dạng đóng**
   của SCM linear-Gaussian đã bị mutilate.

Điểm kỹ thuật cốt lõi (và là chỗ đáng viết nhất): với `do(a = v)`, cột `a` của ma trận trọng số bị
xoá, `c_a = v`, `s2_a = 0`, nên

```
mu    = (I − Bᵀ)⁻¹ c          Sigma = (I − Bᵀ)⁻¹ diag(s2) (I − Bᵀ)⁻ᵀ
```

tính được **chính xác**, và log-likelihood của mẫu can thiệp được đánh giá trên các toạ độ `≠ a`.
Đây chính là thứ phân biệt được hai DAG Markov-tương đương — điều dữ liệu quan sát không làm được.
`tests/test_active.py::test_true_dag_wins_on_interventional_likelihood` kiểm chứng: DAG thật
thắng **100%** số lần thử.

### Các arm

| Nhóm | Arm | Nội dung |
|---|---|---|
| baseline | `oracle` | trần benchmark |
| | `pc_greedy` | baseline cổ điển của repo gốc (không Meek closure) |
| | `pc_greedy_meek` | như trên + Meek closure (baseline **công bằng hơn**) |
| | `llm_e2e` | agent LLM nguyên bản |
| **ours** | **`probe`** | llm_repair ∪ pc_mec, BIC posterior, EIG, Bayes update |
| không gian giả thuyết | `probe_repair_only` | chỉ skeleton do LLM sửa |
| | `probe_llm_graphs` | LLM bịa cả đồ thị (cách dùng LLM ngây thơ) |
| | `probe_skel_only` | mọi định hướng acyclic của skeleton PC — **không LLM**, tách bạch đóng góp của LLM |
| | `probe_mec_only` | MEC của CPDAG từ PC — **không LLM** |
| | `probe_random_hyp` | DAG ngẫu nhiên — sàn chất lượng không gian |
| tầng quyết định | `probe_random_sel` | chọn ngẫu nhiên thay vì EIG |
| | `probe_maxdeg_sel` | chọn theo bậc mơ hồ nhất |
| | `probe_no_bic` | posterior đều, không chấm BIC |
| | `probe_no_update` | không cập nhật Bayes sau can thiệp |
| | `probe_marginal` | nộp edge marginal thay vì MAP |

### Kết quả sơ bộ (d = 6 & 8, `n_obs` = 60, 8 instance — đã chạy thật, qwen3-coder-30b)

| arm | directed F1 | truth ∈ H | best F1 in H |
|---|:-:|:-:|:-:|
| `oracle` | 1.000 | — | — |
| **`probe`** | **0.926** | 0.38 | 0.935 |
| `probe_skel_only` (không LLM) | 0.901 | 0.25 | 0.918 |
| `probe_no_bic` | 0.891 | 0.25 | 0.927 |
| `probe_maxdeg_sel` | 0.882 | 0.38 | 0.909 |
| `pc_greedy_meek` | 0.849 | — | — |
| `probe_mec_only` (không LLM) | 0.849 | 0.25 | 0.849 |
| `pc_greedy` | 0.845 | — | — |
| `probe_random_sel` | 0.813 | 0.38 | 0.935 |
| `probe_llm_graphs` | 0.650 | 0.00 | 0.673 |
| `probe_no_update` | 0.637 | 0.25 | 0.927 |
| `probe_random_hyp` | 0.234 | 0.00 | 0.417 |
| `llm_e2e` | 0.171 | — | — |

> **Ba thông điệp:**
> 1. NemChua (0.926) vượt cả baseline cổ điển (0.845) lẫn LLM agent (0.171), **và rẻ hơn ~7× token**
>    so với `llm_e2e` (979 vs 6717 prompt token/episode).
> 2. **Mọi ablation đều tụt** — EIG, cập nhật Bayes, chấm BIC, chất lượng không gian giả thuyết,
>    mỗi thứ đều đóng góp đo được.
> 3. **Cách hỏi LLM quan trọng hơn việc có dùng LLM hay không**: bảo nó *kiểm toán skeleton*
>    thì được +0.025 so với không dùng LLM; bảo nó *bịa cả đồ thị* thì mất −0.25.

### Phát hiện phụ đáng viết: hai model "chỉnh sửa" khác nhau

`gpt-4o-mini` sửa rất mạnh tay (trung bình 3.5 xoá + 3.1 thêm, gần chạm trần 4/4), `qwen3-coder-30b`
dè dặt hơn (2.8 + 2.6). Cột `repair_remove` / `repair_add` định lượng chính xác chuyện này, và
đó là lý do NemChua có guard "một nửa ngân sách giả thuyết luôn dành cho skeleton gốc của PC" —
để LLM chỉ có thể **thêm** giả thuyết, không bao giờ đẩy giả thuyết mặc định ra ngoài.

### Ablation/analysis có sẵn

- quét `n_obs` ∈ {40, 60, 120, 300, 1000} — **ablation chốt bài**: LLM proposer chỉ đáng tiền
  đúng ở vùng dữ liệu khan, nơi skeleton của PC không đáng tin
- quét `--max-skeleton-edits` ∈ {2, 4, 8} — cho LLM sửa bao nhiêu là vừa
- quỹ đạo entropy posterior theo từng bước (`fig_entropy.png`)
- `truth_rank_initial` / `truth_rank_final` — thứ hạng của DAG thật trong posterior, trước và sau
- token in/out/cached, cost USD, số call, latency, `f1_per_1k_tokens`
- scaling theo `d` = 4 … 12

---

## Vì sao hai bài này bổ trợ nhau

Study 1 là bài **chẩn đoán**: nó *đo* được rằng khâu suy luận mới là nút thắt.
Study 2 là bài **phương pháp**: nó *sửa* đúng nút thắt đó bằng cách chuyển suy luận sang code
và giữ LLM ở vai trò sinh giả thuyết. Nộp hai workshop khác nhau vẫn đứng độc lập được,
nhưng nếu nộp cùng venue thì Study 1 chính là phần motivation của Study 2.

## Giới hạn cần thừa nhận trong paper

- Chỉ linear-Gaussian, không confounder ẩn, quan sát đầy đủ (kế thừa từ benchmark gốc).
- `d ≤ 12`; liệt kê MEC chính xác không scale lên đồ thị lớn (code có đường lấy mẫu dự phòng).
- Giá trị can thiệp cố định `mean + 3·sd`, chưa tối ưu hoá giá trị — chỉ tối ưu hoá *target*.
- Hai model nhẹ; chưa biết kết luận có giữ với model biên giới không (Study 1 gợi ý gap suy luận
  sẽ hẹp lại nhưng không biến mất — số của repo gốc cho GPT-5.5 ủng hộ điều đó).
