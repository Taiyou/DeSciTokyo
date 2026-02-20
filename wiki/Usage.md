# Usage

## インストール

### 前提条件
- Python 3.11 以上
- pip

### セットアップ

```bash
git clone https://github.com/<your-username>/SciOps-Sim.git
cd SciOps-Sim

# 通常インストール
pip install -e .

# 開発用（テストツール含む）
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t sciops-sim .
docker run -v $(pwd)/results:/app/results sciops-sim
```

---

## テスト実行

```bash
# 全テスト（60テスト）
pytest tests/ -v

# ユニットテストのみ（36テスト、約1秒）
pytest tests/unit/ -v

# 統合テストのみ（21テスト、約3秒）
pytest tests/integration/ -v

# 検証テストのみ（3テスト、約1秒）
pytest tests/validation/ -v
```

---

## 実験の実行

### 小規模テスト（動作確認）

```bash
python scripts/run_experiment.py \
  --n-seeds 10 \
  --num-steps 50 \
  --output-dir results_test \
  --no-checkpoint
```
約30秒で完了。

### 中規模実験

```bash
python scripts/run_experiment.py \
  --n-seeds 50 \
  --num-steps 200 \
  --output-dir results_medium
```
36,750回、約2分。

### 本実験（論文用）

```bash
python scripts/run_experiment.py \
  --n-seeds 500 \
  --num-steps 200 \
  --output-dir results_full
```
367,500回、約12分（Apple Silicon 8コア）。チェックポイント付きで中断・再開可能。

### Sobol 感度分析

```bash
python scripts/run_experiment.py \
  --sobol \
  --sobol-n 256 \
  --output-dir results_full
```
76,800回、約3分。

### 結果分析のみ（既存結果を再分析）

```bash
python scripts/run_experiment.py \
  --analyze-only \
  --output-dir results_full
```

---

## CLI オプション一覧

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--output-dir` | `results/` | 出力ディレクトリ |
| `--n-seeds` | 500 | 各条件のランダムシード数 |
| `--n-jobs` | -1 | 並列ワーカー数（-1=全CPU） |
| `--num-steps` | 200 | シミュレーションステップ数 |
| `--batch-size` | 5000 | バッチサイズ |
| `--sobol` | false | Sobol感度分析を実行 |
| `--sobol-n` | 1024 | Sobolサンプルサイズ N |
| `--sobol-strategy` | `ai_sciops` | Sobol対象の戦略 |
| `--analyze-only` | false | 分析と可視化のみ実行 |
| `--timeseries` | false | 時系列データを保存 |
| `--no-checkpoint` | false | チェックポイントを無効化 |

---

## 出力ファイル

```
results_full/
├── scalar_results.csv           # 全結果（367,500行）
├── checkpoints/
│   └── completed_keys.json      # チェックポイント状態
├── sobol/
│   └── sobol_indices_ai_sciops.json  # Sobol指数
└── plots/
    ├── S*_strategy_comparison_a0.png  # 戦略比較棒グラフ（5枚）
    ├── S*_alpha_heatmap.png           # αxストラテジーヒートマップ（5枚）
    ├── S*_phase_diagram.png           # 相図（5枚）
    ├── S*_challenge_amplification.png # 課題増幅曲線（5枚）
    ├── S*_quality_drift.png           # 品質ドリフト（5枚）
    └── sobol_tornado.png              # 感度分析トルネード図
```

---

## Pythonからの利用

```python
from sciops.experiment.runner import execute_single_run
from sciops.experiment.run_config import RunConfig
from sciops.experiment.scenarios import build_scenarios

# シナリオ取得
scenarios = build_scenarios()
scenario = scenarios["S2_alpha_continuous"]

# 単一ランの実行
config = RunConfig(
    scenario_name="S2_alpha_continuous",
    strategy_name="kanban",
    alpha=0.5,
    seed=42,
    pipeline_config=scenario.pipeline_config,
    num_steps=200,
)
result, timeseries = execute_single_run(config, collect_timeseries=True)

print(f"Output: {result.cumulative_output}")
print(f"Net Output: {result.net_output:.2f}")
print(f"Quality: {result.quality_true_mean:.3f}")
print(f"Bottleneck: {result.bottleneck_stage}")
```
