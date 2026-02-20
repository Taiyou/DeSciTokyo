# SciOps-Sim Wiki

**Science as a Production System: Simulating Management Strategies for the Scientific Pipeline and the Emerging Challenges When AI Exceeds Human Capacity**

---

## Overview

SciOps-Sim は、科学研究のプロセス（Survey → Hypothesis → Experiment ⇌ Analysis → Writing → Review → Output）を生産システムとして形式化し、産業管理理論（TOC, PDCA, Kanban, Agile）およびAI最適化戦略の適用効果をシミュレーションで体系的に比較する研究プロジェクトです。

### 核心的な問い

1. **管理理論の適用可能性**: 産業界で発展した生産管理理論は、科学研究パイプラインの生産性を有意に向上させるか？
2. **AIが人間を超えた時の構造的転換**: AIの能力が人間の処理能力を超えた場合、科学研究パイプラインの課題構造はどのように変容するか？

### 実験規模

- **367,500回** のメインシミュレーション（5シナリオ × 7戦略 × 21 α値 × 500シード）
- **76,800回** のSobol感度分析
- **26枚** の可視化プロット

### 主要結果
![S2: 相転移ダイアグラム](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_phase_diagram.png)
*α ≈ 0.55 で AI-Assisted が Kanban を追い越す相転移が確認された*

---

## Wiki 目次

| ページ | 内容 |
|--------|------|
| [Architecture](Architecture) | システム設計・モジュール構成 |
| [Pipeline-Model](Pipeline-Model) | 科学パイプラインモデルの詳細 |
| [Strategies](Strategies) | 7つの管理戦略の設計と実装 |
| [Challenge-Models](Challenge-Models) | 6つの新興課題モデル（Goodhart効果等） |
| [Experiment-Design](Experiment-Design) | 実験設計・シナリオ・統計手法 |
| [Results](Results) | 実験結果の総括（RQ1〜RQ5） |
| [Usage](Usage) | インストール・実行方法 |

---

## Quick Start

```bash
# インストール
pip install -e ".[dev]"

# テスト実行
pytest tests/ -v

# 小規模実験（検証用）
python scripts/run_experiment.py --n-seeds 10 --num-steps 50 --output-dir results_test

# 本実験（367,500回）
python scripts/run_experiment.py --n-seeds 500 --output-dir results_full

# Sobol感度分析
python scripts/run_experiment.py --sobol --output-dir results_full
```
