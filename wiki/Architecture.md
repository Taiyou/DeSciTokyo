# Architecture

## モジュール構成

```
src/sciops/
├── pipeline/          # パイプラインモデル（基盤層）
│   ├── config.py          # StageName, StageConfig, PipelineConfig
│   ├── research_unit.py   # ResearchUnit（研究ユニット）
│   ├── state.py           # PipelineState（実行時状態）
│   └── ai_capability.py   # AI能力の連続パラメータ化（α）
│
├── engine/            # シミュレーションエンジン
│   ├── engine.py          # SimulationEngine（tick-based）
│   ├── actions.py         # 統一操作インターフェース（5種）
│   └── overhead.py        # 管理オーバーヘッドモデル
│
├── strategies/        # 管理戦略（7種）
│   ├── base.py            # Strategy ABC
│   ├── baseline.py        # 管理なし
│   ├── toc_pdca.py        # TOC + PDCA
│   ├── kanban.py          # Kanban
│   ├── agile.py           # Agile Sprint
│   ├── ai_assisted.py     # AI支援付き管理
│   ├── ai_sciops.py       # AI自律最適化（UCB1）
│   ├── oracle.py          # 理論的上界
│   └── factory.py         # 戦略ファクトリ
│
├── metrics/           # 課題モデル・メトリクス
│   ├── challenges.py      # 6つの新興課題モデル
│   └── collector.py       # TickMetrics / MetricsCollector
│
├── experiment/        # 実験フレームワーク
│   ├── scenarios.py       # 5つのシナリオ定義
│   ├── run_config.py      # RunConfig（不変）
│   ├── runner.py          # execute_single_run()
│   ├── parallel.py        # joblib並列実行 + チェックポイント
│   └── sobol.py           # Sobol感度分析（SALib）
│
├── analysis/          # 統計分析
│   ├── statistics.py      # 仮説検定・相転移検出
│   └── aggregation.py     # 集計・ピボットテーブル
│
├── io/                # 入出力
│   ├── results.py         # RunResult / ResultsStore
│   ├── checkpoint.py      # CheckpointManager
│   └── config_loader.py   # YAML設定読込
│
└── visualization/     # 可視化
    └── plots.py           # 8種のプロット関数
```

## 依存関係の流れ

```
pipeline/ ← engine/ ← strategies/
    ↑          ↑           ↑
    |          |           |
metrics/ ← experiment/ ──→ io/
                ↓
           analysis/ → visualization/
```

## 設計原則

### 1. 不変な設定、可変な状態
- `PipelineConfig`, `StageConfig`, `RunConfig` → `frozen=True`（不変）
- `PipelineState`, `StageState` → 可変（シミュレーション中に更新）

### 2. 統一操作インターフェース
全7戦略が同じ5つの操作のみを使用可能:
1. `AllocateResources` — リソース配分
2. `AdjustAILevel` — AI支援レベル変更
3. `InvestUncertaintyReduction` — 不確実性削減投資
4. `AdjustWIPLimit` — WIP制限設定
5. `Restructure` — 構造変更（高コスト）

### 3. 並列安全
- `execute_single_run()` は共有可変状態を持たない
- joblib による並列実行に対応
- チェックポイント/リジュームでバッチ実行を管理

### 4. 再現性
- 全ランダム状態は `np.random.default_rng(seed)` で制御
- 同一シードで完全に同一の結果を保証（検証テスト済み）
