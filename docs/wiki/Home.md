# DeSciTokyo - AI駆動型科学プロセス最適化 PoC

## プロジェクト概要

本プロジェクトは、論文「**Science in the Loop: AI時代における科学研究のプロセスマネジメントと最適化**」の主張をシミュレーション実験で検証するProof of Concept (PoC)です。

### 背景となる問い

> 産業界で長年培われてきたプロセスマネジメント理論（PDCA、アジャイル、制約理論）は、科学研究にも適用できるのか？さらに、AIがこれらの最適化を自律的に行うことで、研究のスループットはどの程度向上するのか？

### プロジェクト全体像

```mermaid
graph TB
    subgraph 論文["📄 論文: Science in the Loop"]
        T1["PDCA / TOC / Agile<br/>産業管理理論"]
        T2["AI for Science<br/>AI駆動型研究"]
        T3["4段階フレームワーク<br/>段階的自律化"]
    end

    subgraph PoC["🔬 PoC シミュレーション"]
        P1["6段階パイプライン<br/>Survey → Review"]
        P2["3つの最適化戦略<br/>Baseline / TOC+PDCA / AI-SciOps"]
        P3["100タイムステップ<br/>メトリクス収集"]
    end

    subgraph Meta["🔄 メタ最適化実験"]
        M1["v4: 管理OH動的最適化<br/>5バリアント比較"]
        M2["v5: AI優越世界<br/>課題構造の変化"]
    end

    subgraph MC["📊 統計的検証"]
        MC1["モンテカルロ N=100<br/>2世界 × 7バリアント"]
        MC2["3世界比較<br/>BN残存世界追加"]
    end

    T1 --> P2
    T2 --> P2
    T3 --> P2
    P1 --> P3
    P2 --> P3
    P3 --> M1
    P3 --> M2
    M1 --> MC1
    M2 --> MC1
    MC1 --> MC2

    style 論文 fill:#e8f4f8,stroke:#2196F3
    style PoC fill:#fff3e0,stroke:#FF9800
    style Meta fill:#f3e5f5,stroke:#9C27B0
    style MC fill:#e8f5e9,stroke:#4CAF50
```

## 主要な発見

### 基本実験: AI-SciOpsが+36.6%の改善を達成

| 戦略 | 総研究出力 | 改善率 |
|------|-----------|--------|
| Baseline（管理なし） | 50.86 | - |
| TOC + PDCA（産業的管理手法） | 54.23 | +6.6% |
| AI-SciOps（AI自律最適化） | 69.48 | **+36.6%** |

→ 詳細: [結果の詳細解釈](./Results-Analysis.md)

### メタ最適化: 管理コスト自体のAI調整

5つの課題バリアント（Oracle、Noisy、Delayed、Recursive、TrustDecay）でAIが管理オーバーヘッドを動的調整。→ 詳細: [管理コスト自体のAI最適化](./Meta-Overhead-Analysis.md)

### モンテカルロ検証: 単一シードの結論を修正

100シードの統計検証により、**seed=42の結論が部分的に覆った**。

```mermaid
graph LR
    subgraph WRONG["❌ seed=42 の結論"]
        W1["現在の世界:<br/>TrustDecay最良 (74.5)"]
    end
    subgraph RIGHT["✅ N=100 の結論"]
        R1["現在の世界:<br/>Oracle最良 (72.8)<br/>上位3つは区別不能"]
    end
    WRONG ==> RIGHT
    style WRONG fill:#ffcdd2,stroke:#d32f2f
    style RIGHT fill:#c8e6c9,stroke:#388E3C
```

→ 詳細: [モンテカルロ実験](./Monte-Carlo-Analysis.md)

### 3世界比較: AI能力 vs ボトルネック撤廃の分解

| 世界 | AI能力 | レビューBN | 最良バリアント | 平均出力 |
|---|---|---|---|---|
| 現在の世界 | 通常 | あり | Oracle (29%) | 72.8 |
| BN残存世界 | 高い | あり | TrustDecay (95%) | 129.9 |
| AI優越世界 | 高い | なし | TrustDecay (100%) | 224.6 |

> **TrustDecayの優位はボトルネック撤廃ではなくAI能力の水準で決まる**

→ 詳細: [ボトルネック残存世界の分析](./Bottleneck-Persists-Analysis.md)

### 結果の視覚的サマリ

```mermaid
xychart-beta
    title "3世界 × 最良バリアントの平均出力 (N=100 seeds)"
    x-axis ["Current:Oracle", "BNP:TrustDecay", "Superior:TrustDecay"]
    y-axis "平均総研究出力" 0 --> 240
    bar [72.8, 129.9, 224.6]
```

## Wiki 目次

### 基本実験
1. **[実験の詳細設計](./Experiment-Design.md)** - パイプラインモデル、パラメータ設計、各条件の詳細
2. **[コードアーキテクチャ](./Architecture.md)** - ソースコードの構造と各モジュールの役割
3. **[結果の詳細解釈](./Results-Analysis.md)** - 6つの可視化図の詳しい読み方と発見
4. **[論文との対応関係](./Paper-Mapping.md)** - シミュレーションの各要素が論文のどの議論に対応するか
5. **[今後の発展](./Future-Work.md)** - このPoCを発展させる方向性

### メタ最適化実験（v4/v5）
6. **[管理コスト自体のAI最適化](./Meta-Overhead-Analysis.md)** - AIが管理OHを動的調整する際の5つの課題をシミュレーション検証
7. **[AI優越世界での課題変化](./AI-Superior-World-Analysis.md)** - AIが人間を上回る場合に課題構造がどう変わるか

### 手法単独比較（v6）
10. **[Individual Methodology Comparison](./Methodology-Comparison.md)** - Baseline, TOC, PDCA, Agile, Kanban, AI-SciOpsを各手法単独で実施した比較実験

### 統計的検証（モンテカルロ）
8. **[モンテカルロ実験](./Monte-Carlo-Analysis.md)** - 100シードによる統計的検証。単一シードの結論が部分的に覆った重要な結果
9. **[ボトルネック残存世界の分析](./Bottleneck-Persists-Analysis.md)** - AIが優秀でも人間レビューが必須な場合の3世界比較

### モデル精緻化・感度分析（v7/v8）
16. **[確率モデルの精緻化](./Probabilistic-Model-Analysis.md)** - ベータ分布・ポアソン過程・プロセス間相関でモデルを精緻化。ランキング頑健性を確認
17. **[パラメータ感度分析](./Sensitivity-Analysis.md)** - 4パラメータの系統的スイープ。TrustDecay/Oracle閾値を発見（AI能力~0.65）

### メタサイエンス実験（MS1-MS4）
11. **[メタサイエンス概要](./Meta-Science-Overview.md)** - 4つのメタサイエンス実験の全体像とアーキテクチャ
12. **[MS1: 出版バイアスと偽陽性蓄積](./Meta-Science-MS1.md)** - p-hackingが偽陽性率を3.4倍に増加
13. **[MS2: 再現性危機ダイナミクス](./Meta-Science-MS2.md)** - AI再現ボットが最も効果的（Truth Ratio: 0.870）
14. **[MS3: 資金配分メカニズム比較](./Meta-Science-MS3.md)** - 4つの資金配分戦略の比較
15. **[MS4: オープンサイエンス](./Meta-Science-MS4.md)** - 知識ネットワーク効果とフリーライダー問題

## クイックスタート

```bash
cd poc/src
pip install matplotlib numpy

# 基本実験
python simulator.py           # 基本シミュレーション
python visualize.py           # 可視化生成

# メタ最適化実験
python run_meta_overhead.py   # v4: 管理OH最適化
python run_ai_superior.py     # v5: AI優越世界

# 手法単独比較
python run_methodology_comparison.py  # v6: 6手法を単独で比較

# 統計的検証
python run_monte_carlo.py             # 2世界モンテカルロ (N=100)
python run_monte_carlo_3worlds.py     # 3世界モンテカルロ (N=100)

# メタサイエンス実験
python run_ms1_publication_bias.py    # MS1: 出版バイアス
python run_ms2_replication_crisis.py  # MS2: 再現性危機
python run_ms3_funding_comparison.py  # MS3: 資金配分
python run_ms4_open_science.py        # MS4: オープンサイエンス

# モデル精緻化・感度分析
python run_probabilistic.py           # v7: 確率モデル精緻化（Beta分布/Poisson/相関）
python run_sensitivity.py             # v8: パラメータ感度分析
```

## 実験の系譜

```mermaid
graph LR
    V1["v1: 基本3戦略"] --> V2["v2: 管理OH導入"]
    V2 --> V3["v3: 高度な戦略追加"]
    V3 --> V4["v4: メタAI最適化<br/>5バリアント"]
    V4 --> V5["v5: AI優越世界"]
    V5 --> V6["v6: 手法単独比較<br/>6手法独立実行"]
    V5 --> MC["MC: 統計的検証<br/>N=100 seeds"]
    MC --> W3["3世界比較<br/>BN残存世界"]
    W3 --> MS["メタサイエンス<br/>MS1-MS4"]
    MS --> V7["v7: 確率モデル<br/>Beta/Poisson/相関"]
    MS --> V8["v8: パラメータ感度<br/>4次元スイープ"]

    style V1 fill:#e3f2fd
    style V4 fill:#f3e5f5
    style MC fill:#e8f5e9
    style W3 fill:#fff8e1
    style MS fill:#fce4ec
    style V7 fill:#e0f7fa
    style V8 fill:#f1f8e9
```
