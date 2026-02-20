# Pipeline Model

## 基本構造：フィードバック付きパイプライン

```
Survey → Hypothesis → Experiment ⇌ Analysis → Writing → Review → Output
            ↑                                              |
            +------- リジェクト時の仮説再検討 ←--------------+
```

### 3つの確率的フィードバック

| フィードバック | 確率 | 経路 |
|--------------|------|------|
| 実験-分析ループ | `p_revision = 0.20` | Analysis → Experiment |
| マイナーリビジョン | `p_minor_revision = 0.15` | Review → Writing |
| メジャーリジェクト | `p_major_rejection = 0.05` | Review → Hypothesis |

ループカウンターが `max_loops = 5` を超えると研究ユニットは打ち切り（断念）。

## 各プロセスのパラメータ

| プロセス | base_throughput | uncertainty | failure_rate | ai_automatable | human_review_needed |
|---------|----------------|-------------|-------------|----------------|-------------------|
| Survey | 2.0 | 0.2 | 0.05 | 0.8 | 0.2 |
| Hypothesis | 1.5 | 0.4 | 0.1 | 0.6 | 0.5 |
| Experiment | 0.8 | 0.5 | 0.15 | 0.3 | 0.3 |
| Analysis | 1.8 | 0.3 | 0.08 | 0.9 | 0.4 |
| Writing | 1.2 | 0.3 | 0.05 | 0.7 | 0.6 |
| Review | 0.6 | 0.2 | 0.1 | 0.4 | 0.8 |

## 有効スループットの計算

各ステージの1ステップあたりの有効スループット:

```
effective_throughput = base_throughput
                      × resources
                      × (1 + ai_level × ai_automatable)     # AI自動化ボーナス
                      × human_factor                         # 人間レビューボトルネック
                      × max(0.1, 1 - |noise × uncertainty|)  # 不確実性ノイズ
```

- `human_factor = (1 - human_review_needed) + human_review_needed × (1 - ai_level × 0.5)`
- `effective_failure_rate = failure_rate × (1 - ai_level × failure_reduction_rate)`

## AI能力の連続パラメータ化

α ∈ [0.0, 1.0] で AI能力を線形補間:

| パラメータ | α=0.0 | α=1.0 |
|-----------|-------|-------|
| Survey.ai_automatable | 0.80 | 0.95 |
| Hypothesis.ai_automatable | 0.60 | 0.90 |
| Experiment.ai_automatable | 0.30 | 0.70 |
| Analysis.ai_automatable | 0.90 | 0.98 |
| Writing.ai_automatable | 0.70 | 0.95 |
| Review.ai_automatable | 0.40 | 0.85 |
| uncertainty_reduction_rate | 0.50 | 0.85 |
| failure_reduction_rate | 0.30 | 0.70 |

`human_review_needed` はαの増加とともに減少（Survey: 0.20→0.02, Review: 0.80→0.10 等）。

## ResearchUnit

各研究ユニットは以下を追跡:

```python
@dataclass
class ResearchUnit:
    id: int                    # 一意識別子
    created_at: int            # 生成タイムステップ
    current_stage: StageName   # 現在のステージ
    quality: float = 1.0       # 真の品質（直接観測不能）
    proxy_quality: float = 1.0 # 代理指標（AI最適化対象）
    human_understanding: float = 1.0  # 人間の理解度
    loop_count: int = 0        # フィードバックループ回数
    total_time: int = 0        # 総所要時間
```

## 到着プロセス

新規研究ユニットはPoisson過程で到着: `n_arrivals ~ Poisson(arrival_rate)`（デフォルト `arrival_rate = 3.0`）。
