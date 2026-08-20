# Chạy trên server — hướng dẫn từng bước

Quy trình: **push từ máy local → pull trên server → tạo env → chạy → kéo kết quả về.**

---

## 0. Trên máy local: push lên GitHub

`.env` đã nằm trong `.gitignore` nên **key sẽ không bị đẩy lên**. Kiểm tra lại cho chắc:

```bash
cd ~/Active-Causal-Discovery-Bench
git check-ignore -v .env          # phải in ra dòng khớp .gitignore
git status --short                # KHÔNG được thấy .env trong danh sách
```

Rồi push:

```bash
git add -A
git commit -m "Add active-experiment studies: selection/inference decomposition + NemChua"
git push origin main
```

> Nếu repo chưa có remote: `git remote add origin git@github.com:<user>/<repo>.git`

---

## 1. Trên server: pull về

```bash
git clone git@github.com:<user>/<repo>.git
cd <repo>
# hoặc nếu đã clone rồi:
git pull origin main
```

---

## 2. Tạo file `.env` **trên server** (không bao giờ commit file này)

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-...........
EOF
chmod 600 .env
```

Code chấp nhận cả `OPENROUTER_API_KEY` lẫn `OPENAI_API_KEY` (repo gốc đang để key OpenRouter
dưới tên `OPENAI_API_KEY`, nên copy nguyên file `.env` cũ sang cũng chạy được).

---

## 3. Tạo conda env

```bash
bash scripts/setup_env.sh
```

Script này sẽ: tạo/cập nhật env `acdb-active` từ `environment.yml`, kiểm tra import,
chạy thử PC trên một instance, xác nhận đọc được API key, và chạy toàn bộ test offline.

Nếu server chưa có conda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash
```

Kích hoạt:

```bash
conda activate acdb-active
```

---

## 4. Chạy thử trước (bắt buộc — tốn ~2 phút, ~$0.03)

```bash
bash scripts/study1.sh smoke
bash scripts/study2.sh smoke
```

Xong phải thấy `study1/smoke/analysis/tables.md` và `traces/study2/smoke/analysis/tables.md`.
Nếu smoke chạy được thì `main` chắc chắn chạy được.

> Study 1 ghi kết quả vào `study1/<run>/`, study 2 vào `traces/study2/<run>/`. Đổi gốc của
> study 1 bằng biến môi trường `ACDB_OUT_ROOT` nếu cần.

---

## 5. Chạy thật

Dùng `tmux` (hoặc `screen`) để không mất tiến trình khi rớt SSH:

```bash
tmux new -s acdb
conda activate acdb-active

bash scripts/study1.sh main    2>&1 | tee logs/study1_main.log
bash scripts/study2.sh main    2>&1 | tee logs/study2_main.log
bash scripts/study1.sh ablation 2>&1 | tee logs/study1_abl.log
bash scripts/study2.sh ablation 2>&1 | tee logs/study2_abl.log
```

Tách khỏi tmux: `Ctrl-b` rồi `d`. Quay lại: `tmux attach -t acdb`.

Hoặc chạy tất cả trong một lần:

```bash
mkdir -p logs
nohup bash -c 'bash scripts/study1.sh all && bash scripts/study2.sh all' > logs/all.log 2>&1 &
tail -f logs/all.log
```

## 5b. Ba stage bổ sung cho bản revision của RauMa

Ba stage này ghép cặp với `main` qua `--seed-map-from study1/main/run_manifest.json`.
**Manifest đó đã nằm sẵn trong repo**, nên không cần chạy lại `main` trên server — chỉ cần
`git pull` là chạy được ngay.

| Lệnh | Episode | Thời gian | Chi phí |
|---|---:|---|---|
| `bash scripts/study1.sh verifier` | 1 440 | ~3 phút | **$0** (không gọi model) |
| `bash scripts/study1.sh local` | 240 | ~15 phút | ~$0.10 |
| `bash scripts/study1.sh rawevidence-paired` | 120 | ~20 phút | ~$0.6 |

Treo cả ba lên tmux bằng một lệnh duy nhất (chạy được kể cả khi SSH rớt):

```bash
cd ~/Active-Causal-Discovery-Bench && git pull origin main && mkdir -p logs

tmux new -d -s rauma "bash -lc '
  cd ~/Active-Causal-Discovery-Bench &&
  bash scripts/study1.sh verifier            2>&1 | tee logs/verifier.log &&
  bash scripts/study1.sh local               2>&1 | tee logs/local.log &&
  bash scripts/study1.sh rawevidence-paired  2>&1 | tee logs/rawpaired.log
  echo \"=== DONE rc=\$? \$(date) ===\"; exec bash'"
```

- Xem tiến độ: `tmux attach -t rauma` — thoát ra bằng `Ctrl-b` rồi `d`
- Xem log không cần attach: `tail -f logs/local.log`
- Liếc nhanh: `tmux capture-pane -pt rauma | tail -20`
- Dừng: `tmux kill-session -t rauma`

Dùng `bash -lc` để tmux đọc `~/.bashrc` và thấy `conda`; `exec bash` giữ pane sống sau khi
chạy xong để còn đọc được output. Mọi stage đều `--resume`, nên kill giữa chừng rồi chạy lại
là nó đi tiếp từ chỗ dở, không tính tiền lại phần đã xong.

### Lấy kết quả về

```bash
# trên máy local
rsync -av --exclude='events.jsonl' --exclude='checkpoint.json' \
    <user>@<server>:~/Active-Causal-Discovery-Bench/study1/ ./study1/
rsync -av <user>@<server>:~/Active-Causal-Discovery-Bench/figures/rauma_f6_verifier.* ./figures/
```

Bảng tóm tắt để dán lại cho reviewer:

```bash
conda run -n acdb-active python -c "
import pandas as pd, glob
d = pd.concat([pd.read_csv(f) for f in glob.glob('study1/localreadout/*/decisions.csv')])
print(d.groupby(['model_tag','prompt'], dropna=False).agg(
    n=('correct','size'), acc=('correct','mean'),
    abstain=('decision', lambda s: (s=='abstain').mean())).round(3).to_string())
"
```

---

## 5c. Ba stage bổ sung cho bản revision của NemChua (study 2)

Ba stage này trả lời ba điểm chính của review. Hai trong ba **không gọi API lần nào**:
chúng replay lại đúng những proposal đã ghi trong `events.jsonl`.

| Lệnh | Episode | Thời gian (24 workers) | Chi phí | Trả lời điều gì |
|---|---:|---|---|---|
| `bash scripts/study2b.sh ranker` | 660 | ~5 phút | **$0** | Hai bậc thang mới: ranker thuần dữ liệu + skeleton thật |
| `bash scripts/study2b.sh sepset` | 1 620 + 60 300 | ~45 phút | ~$5 | Prompt sai thống kê có phải nguyên nhân không |
| `bash scripts/study2b.sh perm` | 152 760 | ~50 phút | **$0** | Random control khớp đúng số edit thực tế |

`perm` là stage quan trọng nhất và tốn $0, nên nếu chỉ chạy được một thứ thì chạy nó.

Treo cả ba lên tmux bằng một lệnh duy nhất:

```bash
cd ~/Active-Causal-Discovery-Bench && git pull origin main && mkdir -p logs

tmux new -d -s nemchua "bash -lc '
  cd ~/Active-Causal-Discovery-Bench &&
  conda activate acdb-active &&
  bash scripts/study2b.sh smoke   2>&1 | tee logs/s2b_smoke.log &&
  bash scripts/study2b.sh ranker  2>&1 | tee logs/s2b_ranker.log &&
  bash scripts/study2b.sh sepset  2>&1 | tee logs/s2b_sepset.log &&
  ACDB_PERM_WORKERS=24 bash scripts/study2b.sh perm 2>&1 | tee logs/s2b_perm.log
  echo \"=== DONE rc=\$? \$(date) ===\"; exec bash'"
```

- Xem tiến độ: `tmux attach -t nemchua` — thoát bằng `Ctrl-b` rồi `d`
- Liếc nhanh: `tmux capture-pane -pt nemchua | tail -20`
- Dừng: `tmux kill-session -t nemchua`

`smoke` chạy đầu tiên và `&&` nối chuỗi, nên nếu smoke hỏng thì ba stage sau không chạy và
không tốn tiền. Mọi stage đều `--resume`: kill giữa chừng rồi chạy lại là đi tiếp từ chỗ dở.

Chỉnh số worker bằng biến môi trường (mặc định `ACDB_WORKERS=12`, `ACDB_PERM_WORKERS=24`,
`ACDB_DRAWS=200`). Nếu server ít core thì đặt `ACDB_PERM_WORKERS` bằng số core thật.

### Kết quả nằm ở đâu

`ranker` và `perm` ghi thêm vào các thư mục **đã có** trong `study2_new/`; `sepset` tạo thư
mục mới `study2b/`.

| Đường dẫn | Nội dung |
|---|---|
| `study2_new/*/analysis/permutation.csv` | 1 dòng/(instance × model): `llm_f1`, `random_mean_f1`, `percentile` |
| `study2_new/*/analysis/permutation_draws.csv` | cả 200 draw, để vẽ null histogram |
| `study2_new/*/analysis/edit_audit.csv` | thêm cột `add_spouse`, `add_other`, `chance_spouse` |
| `study2_new/*_fix/episodes.csv` | thêm arm `probe_stat_edits`, `probe_true_skeleton` |
| `study2_new/semantic*/episodes.csv` | như trên (câu hỏi Sachs) |
| `study2b/sepset_n60/`, `study2b/sepset_main/` | A/B prompt sai vs prompt đúng, cột `repair_evidence` |

### Kéo về máy local

```bash
# trên máy local, trong thư mục repo
rsync -av --exclude='checkpoint.json' \
    <user>@<server>:~/Active-Causal-Discovery-Bench/study2b/ ./study2b/

rsync -av --include='*/' --include='analysis/***' --include='episodes.csv' \
    --include='run_manifest.json' --include='events.jsonl' --exclude='*' \
    <user>@<server>:~/Active-Causal-Discovery-Bench/study2_new/ ./study2_new/
```

`events.jsonl` của `study2b` cần giữ lại (chứa proposal của prompt đúng, dùng để audit lại
mà không phải gọi model). Các file `checkpoint.json` thì bỏ được.

### Kiểm tra nhanh sau khi kéo về

```bash
conda run -n acdb-active python -c "
import pandas as pd, glob
for f in sorted(glob.glob('study2_new/*/analysis/permutation.csv')):
    d = pd.read_csv(f)
    g = d.groupby('model_tag').agg(n=('llm_f1','size'), llm=('llm_f1','mean'),
                                   rnd=('random_mean_f1','mean'), pct=('percentile','mean'))
    g['delta'] = g.llm - g.rnd
    print(f.split('/')[1]); print(g.round(4).to_string(), end='\n\n')
"
```

Nếu cột `delta` dương và `pct` lệch rõ khỏi $0.5$ thì proposal của LLM **có** giá trị vượt
số lượng edit của chính nó, và kết luận trung tâm của paper cần viết lại theo hướng đó.
Nếu `pct` bám quanh $0.5$ thì kết luận hiện tại đứng vững và mạnh hơn hẳn.

### Ước lượng thời gian và chi phí

Số episode là **đếm chính xác**. Thời gian/chi phí ngoại suy từ đo thật ở `d = 10`
(level khó nhất của `main`) nên đây là ước lượng **thiên về cao**.

| Lệnh | Số episode | Thời gian (workers=6) | Chi phí |
|---|---:|---|---|
| `study1.sh smoke` | 72 | ~1 phút | ~$0.07 |
| `study1.sh main` | **720** | ~25 phút | ~$0.6 |
| `study1.sh ablation` | ~**1 900** | ~1 giờ | ~$1.5 |
| `study2.sh smoke` | 100 | ~1 phút | ~$0.09 |
| `study2.sh main` | **960** | ~16 phút | ~$0.8 |
| `study2.sh ablation` | ~**2 400** | ~40 phút | ~$1.8 |

`main` của study1 đã đo thật: 720 episode, **$0.508**, 4.1 triệu token. Các con số ablation là
ngoại suy từ đó nên thiên về cao.

Tổng cả hai bài: **~6 000 episode, dưới ~2.5 giờ, dưới ~$5**. Rẻ và nhanh hơn nhiều so với
repo gốc vì hai model đều nhẹ và NemChua chỉ tốn 1 call LLM mỗi episode.

Riêng ablation `tightbudget` là stage quan trọng nhất về mặt khoa học: nó dùng **cùng seed,
cùng đồ thị, cùng dữ liệu quan sát** như `main`, chỉ khác `budget = |I*|` thay vì `|I*| + 1`,
nên ghép cặp trực tiếp với `main` được. Nếu thiếu thời gian, chạy nó trước tiên:

```bash
python run_study1_decompose.py --out-dir traces/study1/ablation_tightbudget --resume \
    --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150 \
    --budget-slack 0 --inferencers meek --no-e2e
```

> Vì rẻ như vậy nên cứ mạnh dạn tăng `--seeds-per-level` lên 15–20 để CI hẹp lại —
> đó là thứ reviewer workshop hay soi. Gấp đôi seed ≈ gấp đôi thời gian và tiền.

Chỉnh tốc độ/độ mạnh bằng biến môi trường:

```bash
ACDB_WORKERS=12 bash scripts/study1.sh main            # nhiều luồng hơn (coi chừng rate limit)
ACDB_MODELS=qwen3-coder-30b bash scripts/study2.sh main # chỉ 1 model cho rẻ
ACDB_ENV=my-env bash scripts/study1.sh main            # tên env khác
```

---

## 6. Đứt giữa chừng thì sao?

Chạy **lại đúng lệnh cũ**. Mọi script đều truyền `--resume`, và runner ghi
`checkpoint.json` sau **từng** episode, nên nó bỏ qua phần đã xong.

```bash
bash scripts/study1.sh main        # tiếp tục chỗ dở
```

Chạy lại riêng những episode đã lỗi:

```bash
python run_study1_decompose.py --out-dir traces/study1/main --resume --retry-failed \
    --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150
```

Xem lỗi là gì:

```bash
python - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("traces/study1/main/episodes.csv")))
bad = [r for r in rows if r["status"] != "success"]
print(f"{len(bad)}/{len(rows)} failed")
for msg, n in collections.Counter(r["error"][:120] for r in bad).most_common(10):
    print(f"{n:5d}  {msg}")
EOF
```

---

## 7. Lấy bảng và hình cho paper

Analysis chạy tự động ở cuối mỗi stage. Chạy lại thủ công lúc nào cũng được:

```bash
python scripts/analyze.py --study 1 --run-dir traces/study1/main
python scripts/analyze.py --study 2 --run-dir traces/study2/main
```

Sản phẩm nằm trong `<run-dir>/analysis/`:

| File | Nội dung |
|---|---|
| `tables.md` | **tất cả bảng ở dạng markdown — copy thẳng vào paper** |
| `t*.csv` | từng bảng dạng CSV |
| `fig_grid.png` | Study 1: lưới selector × inferencer |
| `fig_regret.png` | Study 1: regret chọn thí nghiệm theo từng bước |
| `fig_main.png` | Study 2: NemChua so với baseline và ablation |
| `fig_entropy.png` | Study 2: entropy posterior giảm theo từng thí nghiệm |

Kéo về máy local:

```bash
rsync -avz --include='*/' --include='analysis/**' --include='*.csv' --include='run_manifest.json' \
      --exclude='*' <user>@<server>:~/<repo>/traces/ ./traces_from_server/
```

Muốn kéo cả `events.jsonl` (log đầy đủ từng call LLM: prompt, response, token, cost, latency)
thì bỏ bớt `--exclude`. File này khá lớn nhưng là thứ reviewer có thể dùng để chấm lại.

---

## 8. Dữ liệu thô có những gì (phần cho mục Reproducibility của paper)

Mỗi `--out-dir` chứa:

| File | Nội dung |
|---|---|
| `episodes.csv` | **một dòng / episode** — đủ 4 lớp metric, metric mới, token in/out/cached, cost USD, số call, latency, số lần phải repair JSON |
| `steps.csv` | **một dòng / thí nghiệm** — target, regret, EIG, kích thước MEC, entropy, số cạnh gỡ được, độ chính xác định hướng |
| `events.jsonl` | **mọi call LLM**: model, provider, usage, payload đã parse, lỗi nếu có |
| `run_manifest.json` | toàn bộ CLI args, seed map, spec của từng level — đủ để chạy lại y hệt |
| `checkpoint.json` | trạng thái resume |
| `summary_by_arm*.csv` | tổng hợp mean/sd/CI95 |

Mọi arm dùng **chung seed map** (ghi trong `run_manifest.json`), nên kết quả là **paired theo
instance** — dùng paired t-test hoặc Wilcoxon signed-rank khi so hai arm.

---

## 9. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `RuntimeError: No OpenRouter API key found` | thiếu `.env`, hoặc chạy sai thư mục. Thêm `--env-file /đường/dẫn/.env` |
| Nhiều episode fail với `HTTP 429` | giảm `ACDB_WORKERS` xuống 2–3 rồi `--resume --retry-failed` |
| `HTTP 402` | hết credit OpenRouter |
| `tool call could not be repaired` | model trả sai schema quá 2 lần. Tăng `--max-repairs 4` |
| Chạy chậm bất thường | `--eig-max-members 128` (study 1) hoặc `--max-dags-per-skeleton 512` (study 2) |
| `ModuleNotFoundError: causallearn` | quên `conda activate acdb-active` |
| Kết quả mọi arm giống hệt nhau | PC đã định hướng hết cạnh nên không có thí nghiệm nào chạy — kiểm tra cột `pc_undirected_edges` |

---

## 10. Kiểm tra nhanh sức khỏe của một run đang chạy

```bash
python - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("traces/study2/main/episodes.csv")))
ok = [r for r in rows if r["status"] == "success"]
cost = sum(float(r["cost_usd"] or 0) for r in rows)
tok  = sum(float(r["total_tokens"] or 0) for r in rows)
print(f"{len(ok)}/{len(rows)} thành công | tổng cost ${cost:.3f} | tổng token {tok:,.0f}")
by = collections.defaultdict(list)
for r in ok: by[r["arm"]].append(float(r["directed_f1"]))
for arm, vals in sorted(by.items(), key=lambda x: -sum(x[1])/len(x[1])):
    print(f"  {arm:<20} n={len(vals):3d}  directed_f1={sum(vals)/len(vals):.3f}")
EOF
```
