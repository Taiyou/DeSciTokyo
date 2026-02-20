# Management Strategies

## 統一操作インターフェース

全7戦略が使用可能な操作は以下の5種のみ。戦略間の差異は「どの操作を、いつ、どの順序で実行するか」という意思決定ロジックの差として表現されます。

| 操作 | 説明 |
|------|------|
| `AllocateResources(stage, amount)` | リソース再配分（総量制約内） |
| `AdjustAILevel(stage, delta)` | AI支援レベルの変更 |
| `InvestUncertaintyReduction(stage, amount)` | 不確実性削減への投資（収穫逓減） |
| `AdjustWIPLimit(stage, limit)` | WIP制限の設定・変更 |
| `Restructure(description, cost, downtime)` | パイプライン構造変更（高コスト+ダウンタイム） |

---

## 人間管理手法群（AI支援レベル = 0）

### 1. Baseline（管理なし）
- リソース均等配分、一切の最適化操作なし
- オーバーヘッド: 0
- **結果**: 実質ゼロ出力。管理の必要性を示す対照群

### 2. TOC + PDCA
- **制約理論**でボトルネック特定 → リソース集中 → **PDCAサイクル**で反復改善
- 10ステップ周期でCheck→Plan→Do→Act
- ボトルネックに35%のリソースを集中、残りを均等配分
- 改善が小さい場合は不確実性削減に投資
- オーバーヘッド: base=0.15, human_coord=0.15

### 3. Kanban
- WIP制限の設定（初期値: 各ステージ8）
- プルベースのフロー制御
- 5ステップごとにWIP比率に基づくリソース配分（40%均等 + 60%WIP比例）
- オーバーヘッド: base=0.10, human_coord=0.10
- **実験結果**: 人間管理手法の中で最良（平均出力 43.2）

### 4. Agile Sprint
- 10ステップのスプリント単位で計画・実行・レトロスペクティブ
- スプリントごとにスループットの逆数に比例してリソース再配分（遅いステージに重点）
- 最遅ステージに不確実性削減を投資
- オーバーヘッド: base=0.20, human_coord=0.25（スプリント儀式のコストが最大）

---

## AI関与管理手法群

### 5. AI-Assisted（AI支援付き管理）
- AIによるボトルネック検知でWIP分析を強化
- 5ステップごとにAIレベルを漸増（+0.05/回、上限0.8）
- ボトルネックに30%のリソース集中
- ルールベース（if-then）の意思決定。構造変更なし
- オーバーヘッド: base=0.10, ai_infra=0.15, human_coord=0.05
- **実験結果**: α > 0.55 でKanbanを追い越す相転移。α=1.0 で出力 111.7（+1,217%）

### 6. AI-SciOps（AI自律最適化）
- 全5操作を活用。**UCB1バンディットアルゴリズム**で操作テンプレートを選択
- 5つのテンプレート: ボトルネック集中、AI全面強化、不確実性削減、WIP制限設定、構造変更
- 3ステップごとに操作を決定し、前回操作の報酬を学習
- 構造変更は1回限り（コスト0.5 + 2ステップダウンタイム）
- オーバーヘッド: base=0.10, ai_infra=0.20, human_coord=0.05

### 7. Oracle（理論的上界）
- 全プロセスの状態を完全観測
- 毎ステップでAIレベルを最大化（+0.15/回）
- スループットの逆数に比例する最適リソース配分
- 最高不確実性のステージに投資0.5
- オーバーヘッド: base=0.05, ai_infra=0.30, human_coord=0.02
- **実験結果**: 全シナリオで最高出力（平均 103.3）だが、オーバーヘッドも最大（368単位）

---

## 管理オーバーヘッド比較

| 戦略 | base_cost | ai_infra_cost | human_coord_cost | 実測平均OH | 出力/OH効率 |
|------|-----------|---------------|------------------|----------|-----------|
| Baseline | 0.0 | 0.0 | 0.0 | 0.0 | — |
| TOC+PDCA | 0.15 | 0.0 | 0.15 | 〜30 | 0.14 |
| Kanban | 0.10 | 0.0 | 0.10 | 〜39 | **1.10** |
| Agile | 0.20 | 0.0 | 0.25 | 〜45 | 0.72 |
| AI-Assisted | 0.10 | 0.15 | 0.05 | 〜95 | 0.49 |
| AI-SciOps | 0.10 | 0.20 | 0.05 | 〜117 | 0.43 |
| Oracle | 0.05 | 0.30 | 0.02 | 〜368 | 0.28 |

---

## 戦略比較の可視化

### 戦略別出力比較（α=0、管理能力のみの比較）
![S1: 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_strategy_comparison_a0.png)

### α-戦略ヒートマップ（AI能力の効果）
![S2: α-戦略ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_alpha_heatmap.png)

### 相転移ダイアグラム
![S2: 相転移ダイアグラム](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_phase_diagram.png)
