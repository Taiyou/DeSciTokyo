# コードアーキテクチャ

## ファイル構成

```
poc/
├── README.md                              # プロジェクト概要
├── src/
│   ├── scientific_process.py              # 科学パイプラインモデル
│   ├── optimizers.py                      # 最適化戦略（3種）
│   ├── simulator.py                       # シミュレーションエンジン
│   └── visualize.py                       # 結果の可視化
└── results/
    ├── Baseline_No_Optimization.json      # Baseline結果データ
    ├── TOC_+_PDCA.json                    # TOC+PDCA結果データ
    ├── AI-SciOps_Autonomous_Optimization.json  # AI-SciOps結果データ
    └── figures/
        ├── 01_cumulative_output.png       # 累積研究出力
        ├── 02_system_throughput.png        # システムスループット推移
        ├── 03_bottleneck_analysis.png      # ボトルネック分布
        ├── 04_throughput_heatmap.png       # スループットヒートマップ
        ├── 05_wip_accumulation.png         # WIP蓄積パターン
        └── 06_summary_comparison.png       # 総合比較
```

## モジュール設計

### `scientific_process.py` — パイプラインモデル

科学研究の個々のプロセスをデータクラスとして定義しています。

```
ProcessConfig（設定、不変）
  ├── name, base_throughput, uncertainty, failure_rate
  ├── resource_cost, ai_automatable, human_review_needed
  └── min_throughput, max_throughput

ProcessStep（実行時状態、可変）
  ├── config: ProcessConfig
  ├── throughput, allocated_resources, ai_assistance_level
  ├── work_in_progress, completed_units, failed_units
  ├── rework_units, human_review_backlog
  ├── effective_throughput() → AI支援・リソース・人間ボトルネックを考慮した実効スループット
  └── step(incoming_work) → 1タイムステップの処理を実行、出力を返す
```

**設計意図**: `ProcessConfig`（不変の構造パラメータ）と`ProcessStep`（可変のランタイム状態）を分離することで、最適化戦略が構造パラメータ自体を変更する（Stage 3-4）ケースも扱えるようにしています。

### `optimizers.py` — 最適化戦略

Strategy パターンを使い、3つの最適化アルゴリズムを統一インターフェースで実装しています。

```
Optimizer（抽象基底クラス）
  ├── optimize(pipeline, time_step, total_resources) → pipeline
  └── get_bottleneck(pipeline) → ProcessStep

BaselineOptimizer
  └── optimize(): 均等配分のみ

TOCPDCAOptimizer
  ├── pdca_cycle_length: int = 10
  ├── current_phase: Plan/Do/Check/Act
  └── optimize(): TOCボトルネック特定 + PDCAサイクル

AISciOpsOptimizer
  ├── history: list[dict]           — 過去の状態記録
  ├── exploration_rate: float       — 探索率（0.3→0.05に減衰）
  ├── learned_allocations: dict     — 学習済みリソース配分
  ├── process_pruned: set           — 枝刈り済みプロセス
  ├── _stage1_optimize()            — 人間フィードバック付き
  ├── _stage2_optimize()            — 自律的最適化
  ├── _stage3_optimize()            — プロセス枝刈り
  └── _stage4_optimize()            — メタプロセス再組織化
```

**設計意図**: `OptimizationAction`データクラスで全ての最適化介入を記録し、後から「いつ、どのプロセスに、どのような介入をしたか」を追跡可能にしています。

### `simulator.py` — シミュレーションエンジン

パイプラインとオプティマイザーを組み合わせて実行し、メトリクスを収集します。

```
Simulator
  ├── optimizer: Optimizer
  ├── pipeline: list[ProcessStep]
  ├── run(time_steps) → SimulationResult
  │     各ステップで:
  │     1. optimizer.optimize() でパイプライン調整
  │     2. 各プロセスに作業を流す（直列）
  │     3. メトリクス収集
  └── SimulationResult
        ├── total_output: float
        ├── metrics: list[TimeStepMetrics]
        ├── optimization_actions: list[dict]
        └── final_state: dict

run_experiment()
  └── 3つのoptimizerで同一条件（seed=42）でSimulatorを実行し結果をJSONに保存
```

### `visualize.py` — 可視化

6種類のグラフを生成します：

| 図 | 関数 | 描画内容 |
|----|------|---------|
| 01 | `plot_cumulative_output()` | 3戦略の累積出力を時系列で比較 + Stage境界線 |
| 02 | `plot_system_throughput()` | 瞬時スループットの5ステップ移動平均 |
| 03 | `plot_bottleneck_analysis()` | 各プロセスがボトルネックだった回数の棒グラフ |
| 04 | `plot_process_throughputs_heatmap()` | プロセス×時間のスループットヒートマップ |
| 05 | `plot_wip_accumulation()` | 各プロセスのWIP蓄積を3戦略で比較 |
| 06 | `plot_summary_comparison()` | 最終的な4指標の棒グラフ比較 |

## データフロー

```
[simulator.py]
  │
  ├── create_default_pipeline()  ← scientific_process.py
  ├── BaselineOptimizer()        ← optimizers.py
  ├── TOCPDCAOptimizer()         ← optimizers.py
  ├── AISciOpsOptimizer()        ← optimizers.py
  │
  ├── run() × 3 strategies
  │     └── 100 time steps × 6 processes
  │
  └── save to results/*.json
         │
         └── [visualize.py] → results/figures/*.png
```
