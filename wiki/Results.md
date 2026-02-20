# Results

367,500回のシミュレーションと76,800回のSobol感度分析の結果を、各リサーチクエスチョンに沿って報告します。

---

## RQ1: 管理手法は科学パイプラインの出力を有意に向上させるか？

### 回答: **はい。全シナリオで p < 0.0001**

全シナリオでKruskal-Wallis検定がH > 2000を示し、戦略間に極めて有意な差が確認されました。

### 戦略ランキング（全条件平均 Net Output）

| 順位 | 戦略 | 平均出力 | Oracleとの比 |
|------|------|---------|-------------|
| 1 | Oracle | 103.3 | 100% |
| 2 | AI-SciOps | 50.3 | 48.7% |
| 3 | AI-Assisted | 46.1 | 44.6% |
| 4 | Kanban | 43.2 | 41.8% |
| 5 | Agile | 32.5 | 31.4% |
| 6 | TOC+PDCA | 4.1 | 3.9% |
| 7 | Baseline | 0.0 | 0% |

![S1: 戦略比較 (α=0)](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_strategy_comparison_a0.png)

**主要知見**:
- Baselineは実質ゼロ出力 → **科学パイプラインには管理が不可欠**
- 人間管理手法のみではKanbanが最良（43.2）
- Oracleは2位の約2倍 → **完全情報の価値は極めて大きい**

---

## RQ2: AI自律最適化は管理手法のみと比較して優れるか？

### 回答: **AI能力（α）に強く依存する**

| 戦略 | α=0.0 | α=1.0 | 改善率 |
|------|-------|-------|--------|
| AI-Assisted | 8.5 | 111.7 | **+1,217%** |
| Oracle | 56.1 | 138.1 | +146% |
| AI-SciOps | 41.7 | 58.6 | +40% |
| Kanban | 43.2 | 43.2 | 0% |
| Agile | 32.5 | 32.5 | 0% |

![S2: α-戦略ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_alpha_heatmap.png)

**主要知見**:
- Kanban/Agile/TOC+PDCA は α に**完全に非感応**（Phase Diagramで水平線）
- AI-Assisted は α=1.0 で出力が**12倍以上**に跳ね上がる
- α=0.0（現在のAI能力）では Kanban(43.2) > AI-SciOps(41.7) > AI-Assisted(8.5)
- α=1.0（AI優越世界）では AI-Assisted(111.7) > Oracle以外の全戦略

---

## RQ3: 最適戦略が切り替わる相転移点は存在するか？

### 回答: **はい。α ≈ 0.55 に明確な相転移が存在する**

Phase Diagram（S2シナリオ）の主要な読み取り:

```
α = 0.0:  Oracle(47) >> Kanban(42) > AI-SciOps(39) >> AI-Assisted(1)
α = 0.3:  Oracle(89) >> Kanban(42) > AI-SciOps(42) > AI-Assisted(13)
α = 0.55: Oracle(106) > AI-SciOps(46) > Kanban(42) ≈ AI-Assisted(36) ← 交差点
α = 0.75: Oracle(114) > AI-Assisted(62) > AI-SciOps(51) > Kanban(42)
α = 1.0:  Oracle(121) > AI-Assisted(102) > AI-SciOps(54) > Kanban(42)
```

![S2: 相転移ダイアグラム](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_phase_diagram.png)

**相転移の構造**:
1. **α < 0.55**: Kanban が AI-Assisted を上回る。AIの能力不足により、シンプルな管理が勝る
2. **α ≈ 0.55**: AI-Assisted が Kanban を追い越す**第一の相転移**
3. **α ≈ 0.75**: AI-Assisted が AI-SciOps をも追い越す**第二の相転移**

### S5（ハイリスク研究）の特異的動態
- α=0.55 で AI-Assisted → Oracle への切替
- α=0.75 で Oracle → AI-Assisted への再逆転
- **高不確実性環境では、適応的戦略が完全情報戦略をも上回る場面がある**

![S5: ハイリスクシナリオの相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_phase_diagram.png)

---

## RQ4: 情報論的課題は増幅されるか？

### 回答: **はい。品質ドリフトと理解度低下が確認された**

### Goodhart効果（品質ドリフト）
| 戦略 | 真の品質 | ドリフト | リスク |
|------|---------|---------|--------|
| Kanban | 1.000 | 0.000 | なし |
| Agile | 0.980 | 0.000 | なし |
| AI-SciOps | 0.977 | 0.023 | 低 |
| Oracle | 0.958 | 0.032 | 中 |
| AI-Assisted | 0.923 | 0.031 | 中 |

### αの増加に伴う変化
- **真の品質**: 0.885 → 0.870 へ単調減少（約1.7%低下）
- **品質ドリフト**: αの中域でピーク後、やや安定化（逆U字型）
- **管理オーバーヘッド**: 128 → 113 へ減少（AI効率化の正の効果）

![S2: 品質ドリフト分析](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_quality_drift.png)

![S2: 課題増幅曲線](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_challenge_amplification.png)

**核心的知見**: 伝統的管理（Kanban, Agile）は品質ドリフトがゼロだが出力に天井がある。AI戦略は出力を大幅に伸ばす代わりに、Goodhart効果による品質劣化を受ける。**出力と品質のトレードオフは構造的に不可避**。

---

## RQ5: どのパラメータ仮定が結論に最も影響するか？

### 回答: **パイプラインのボトルネック（Review, Experiment, Writing）が支配的**

Sobol総合指数（ST）による重要度ランキング:

| 順位 | パラメータ | ST | 解釈 |
|------|-----------|------|------|
| 1 | **review_throughput** | **0.354** | 最重要。査読が最大のボトルネック |
| 2 | **experiment_throughput** | **0.252** | 実験処理能力 |
| 3 | **writing_throughput** | **0.213** | 執筆処理能力 |
| 4 | hypothesis_throughput | 0.149 | 仮説構築速度 |
| 5 | p_revision | 0.089 | 実験差戻し確率 |
| 6 | **alpha** | **0.083** | AI能力（14パラメータ中6位） |
| 7 | p_minor_revision | 0.075 | マイナーリビジョン確率 |
| 8-14 | その他 | < 0.05 | 低影響 |

![Sobol 感度分析トルネードチャート](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/sobol_tornado.png)

**主要知見**:
- 上位3パラメータ（Review + Experiment + Writing）で感度の**82%**を占める
- alpha（AI能力）は14パラメータ中**6位**（ST=0.083）— 重要だが支配的ではない
- 情報論的課題パラメータ（Goodhart, 理解度減衰, 新規性減衰）はいずれも ST < 0.011 で**直接効果は小さい**
- **結論の頑健性**: パイプラインのスループット仮定を大きく変えない限り、主要な結論は維持される

---

## 論文ナラティブへの示唆

実験結果は提案書のナラティブを強く支持します:

1. **「科学には管理が必要」**: Baseline ≈ 0 vs Kanban ≈ 43 が決定的な証拠
2. **「AIが進展すると管理課題は消えるのではなく変容する」**: α ≈ 0.55 での相転移とGoodhart効果の増幅が直接的証拠
3. **「投資すべきはAIツールだけでなく、プロセスの観測基盤」**: Review throughput が Sobol 最重要パラメータ（ST=0.354）であり、査読プロセスの改善が最大のレバレッジ

---

## 全シナリオ可視化ギャラリー

### S1: 現在基準シナリオ
| 戦略比較 | αヒートマップ |
|---------|-------------|
| ![S1 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_strategy_comparison_a0.png) | ![S1 ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_alpha_heatmap.png) |

| 相転移 | 品質ドリフト | 課題増幅 |
|--------|-----------|---------|
| ![S1 相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_phase_diagram.png) | ![S1 品質](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_quality_drift.png) | ![S1 課題](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S1_baseline_challenge_amplification.png) |

### S2: AI能力連続変化シナリオ（主要シナリオ）
| 戦略比較 | αヒートマップ |
|---------|-------------|
| ![S2 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_strategy_comparison_a0.png) | ![S2 ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_alpha_heatmap.png) |

| 相転移 | 品質ドリフト | 課題増幅 |
|--------|-----------|---------|
| ![S2 相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_phase_diagram.png) | ![S2 品質](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_quality_drift.png) | ![S2 課題](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_challenge_amplification.png) |

### S3: ボトルネック残存シナリオ
| 戦略比較 | αヒートマップ |
|---------|-------------|
| ![S3 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S3_bottleneck_strategy_comparison_a0.png) | ![S3 ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S3_bottleneck_alpha_heatmap.png) |

| 相転移 | 品質ドリフト | 課題増幅 |
|--------|-----------|---------|
| ![S3 相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S3_bottleneck_phase_diagram.png) | ![S3 品質](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S3_bottleneck_quality_drift.png) | ![S3 課題](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S3_bottleneck_challenge_amplification.png) |

### S4: 理論系ラボシナリオ
| 戦略比較 | αヒートマップ |
|---------|-------------|
| ![S4 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S4_theory_lab_strategy_comparison_a0.png) | ![S4 ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S4_theory_lab_alpha_heatmap.png) |

| 相転移 | 品質ドリフト | 課題増幅 |
|--------|-----------|---------|
| ![S4 相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S4_theory_lab_phase_diagram.png) | ![S4 品質](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S4_theory_lab_quality_drift.png) | ![S4 課題](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S4_theory_lab_challenge_amplification.png) |

### S5: ハイリスク探索研究シナリオ
| 戦略比較 | αヒートマップ |
|---------|-------------|
| ![S5 戦略比較](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_strategy_comparison_a0.png) | ![S5 ヒートマップ](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_alpha_heatmap.png) |

| 相転移 | 品質ドリフト | 課題増幅 |
|--------|-----------|---------|
| ![S5 相転移](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_phase_diagram.png) | ![S5 品質](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_quality_drift.png) | ![S5 課題](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S5_high_risk_challenge_amplification.png) |

### Sobol 感度分析
![Sobol トルネードチャート](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/sobol_tornado.png)
