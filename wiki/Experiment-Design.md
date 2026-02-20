# Experiment Design

## シナリオ設計

| シナリオ | AI能力 α | パイプライン特性 | 目的 |
|---------|---------|----------------|------|
| S1: 現在基準 | 0.0 固定 | デフォルト、フィードバックなし | ベースライン比較 |
| S2: AI能力連続変化 | 0.0–1.0 (21点) | デフォルト、フィードバックあり | **主要シナリオ**: 相転移点の特定 |
| S3: ボトルネック残存 | 0.0–1.0 | Review.human_review_needed 固定 (0.8) | 制度的制約下でのAI効果 |
| S4: 理論系ラボ | 0.0–1.0 | Hypothesis遅延、Experiment高速 | 分野による差異 |
| S5: ハイリスク探索研究 | 0.0–1.0 | 全uncertainty増加、p_reject高 | 革新的研究への適用 |

## 実験条件

- **戦略数**: 7（Baseline, TOC+PDCA, Kanban, Agile, AI-Assisted, AI-SciOps, Oracle）
- **α値**: 0.00, 0.05, 0.10, ..., 1.00（21点）
- **シード数**: 500（各条件）
- **シミュレーションステップ**: 200（= 200週 ≈ 4年）
- **合計実行数**: 5 × 7 × 21 × 500 = **367,500回**

## 統計的検証

| 検証 | 手法 | 目的 |
|------|------|------|
| 多重比較 | Kruskal-Wallis → Dunn の事後検定 + Bonferroni 補正 | 全7戦略の一括比較 |
| 2群間比較 | Welch の t 検定 | 特定の戦略ペアの有意差 |
| 効果量 | Cohen's d | 有意差の実質的大きさ |
| ランク安定性 | Kendall の一致係数 W | 異なるシードでの順位の安定性 |
| 相転移の特定 | α の関数としての戦略順位の変化点検出 | RQ3 |

## Sobol 感度分析

14次元のパラメータ空間に対するSaltelli サンプリング:

| パラメータ | 探索範囲 |
|-----------|---------|
| survey_throughput | [1.0, 4.0] |
| hypothesis_throughput | [0.5, 3.0] |
| experiment_throughput | [0.3, 2.0] |
| analysis_throughput | [0.8, 3.5] |
| writing_throughput | [0.5, 2.5] |
| review_throughput | [0.3, 1.5] |
| p_revision | [0.0, 0.6] |
| p_minor_revision | [0.0, 0.5] |
| p_major_rejection | [0.0, 0.15] |
| arrival_rate | [1.0, 6.0] |
| alpha | [0.0, 1.0] |
| goodhart_drift_rate | [0.0, 0.02] |
| understanding_decay_rate | [0.0, 0.05] |
| novelty_decay_rate | [0.0, 0.03] |

- N=256, D=14 → 約7,680サンプル点
- 各点で10シード → 合計 **76,800回** のシミュレーション
- Sobol 一次指数（S1）と総合指数（ST）を計算

### Sobol 感度分析結果
![Sobol トルネードチャート](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/sobol_tornado.png)

## 並列実行

- **バックエンド**: joblib (loky)
- **バッチサイズ**: 5,000 runs/batch
- **チェックポイント**: バッチ完了ごとに `completed_keys.json` に保存
- **実測性能**: 367,500回 ≈ 約12分（Apple Silicon, 8コア）
