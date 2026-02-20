# Challenge Models

AIが人間の処理能力を超えた場合に構造的に発生する6つの課題をモデル化しています。

## 消滅する課題群

### D. 人間レビューのボトルネック
- α > 0.6 で `human_review_needed` が十分低下し、実質的に解消
- Review ステージの throughput が劇的に改善

### E. 人間信頼の動態（TrustDecayModel）
- AIの失敗で信頼が低下、成功で回復
- `trust -= trust_decay_base × n_failures`
- `trust += trust_recovery_rate × n_successes`
- α > 0.5 でAIが補完し問題化しなくなる

---

## 増幅される課題群（予備実験で発見）

### A. 観測性の問題（ObservabilityGapModel）
```
observation_noise = observability_decay_rate × alpha × ln(1 + timestep)
```
- αが大きいほど、状態観測にノイズが乗る
- 高速システムでは1回の誤推定による機会損失が拡大
- **実験結果**: αの増加に伴い観測ノイズが対数的に増加

### B. 遅延フィードバック（DelaySensitivityModel）
```
delay_cost = delay_ticks ^ (1 + alpha × (exponent - 1))
```
- 高速システムでは同じ遅延ティック数でもコストが超線形的に増大
- `delay_amplification_exponent = 1.5`
- **実験結果**: α=1.0 でのフィードバック遅延コストは α=0.0 の約3.2倍

### C. 自己参照コスト（管理オーバーヘッド）
- メタAIの計算コスト比率が、人間ボトルネック解消とともに相対的に増大
- Oracle のオーバーヘッド 368 単位 vs Kanban の 39 単位（9.4倍）
- **実験結果**: Oracleの出力/OH効率は0.28で最低。情報の完全性にはコストがかかる

---

## 新たに出現する課題群（本実験で探索）

### F. 品質の定義の変容 — Goodhart効果（QualityDriftModel）
```python
# 代理指標は改善するが、真の品質との乖離が蓄積
drift = goodhart_drift_rate × (1 + alpha × goodhart_ai_amplification)
proxy_quality += drift × ai_level      # 代理指標は上昇
quality -= drift × ai_level × 0.1      # 真の品質は微減
```
- AI最適化は `q_proxy` を向上させる一方で `q_true` との乖離が蓄積
- **実験結果**:
  - Kanban/Agile: ドリフト = 0.000（AI非使用のため劣化なし）
  - Oracle: ドリフト = 0.032
  - AI-Assisted: ドリフト = 0.031
  - 真の品質は α の増加とともに 0.885 → 0.870 へ約1.7%低下

### G. 科学的冗長性の加速（RedundancyModel）
```python
novelty_factor = exp(-novelty_decay_rate × cumulative_output)
effective_output = (1 - exp(-k × N)) / k
```
- 累積出力が増えるほど新規性ファクターが指数減衰
- 高速パイプラインほど個別スループットは上がるが、有効出力（新規性のある成果）は収穫逓減
- **実験結果**: novelty_decay_rate の Sobol 総合指数 = 0.011（直接効果は小さいが、αとの交互作用あり）

### H. 科学のブラックボックス化（BlackBoxingModel）
```python
understanding -= understanding_decay_rate × ai_level  # 理解度漸減
resilience = mean(human_understanding)                 # システム回復力
is_fragile = resilience < resilience_threshold (0.3)
```
- AIの自動化率が高いと人間の理解度が漸減
- 理解度が閾値を下回ると修正能力が低下し、システム回復力が劣化
- **実験結果**: AI-Assisted/Oracle のresilienceはα=1.0で低下傾向

---

## 課題の構造的シフト：全体像

```
AI能力 α: 0.0 ──────────────────────────────→ 1.0

消滅する課題:
  D. 人間レビューボトルネック  ████████░░░░ → 解消
  E. 人間信頼の動態           ██████░░░░░░ → 安定化

増幅される課題:
  A. 観測性の問題             ░░░░████████ → 拡大
  B. 遅延フィードバック       ░░░░░██████████ → 超線形増大
  C. 自己参照コスト           ░░░░░░████████ → 相対的重荷増大

新たに出現する課題:
  F. Goodhart効果             ░░░░░░░██████ → 品質乖離蓄積
  G. 科学的冗長性             ░░░░░░░░████ → 新規性減衰
  H. ブラックボックス化       ░░░░░░░░░███ → 理解喪失
```

## 可視化

### 課題増幅曲線（S2: AI能力連続変化シナリオ）
![S2: 課題増幅曲線](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_challenge_amplification.png)

### 品質ドリフト（Goodhart効果）
![S2: 品質ドリフト](https://raw.githubusercontent.com/Taiyou/sciops-sim/main/docs/images/S2_alpha_continuous_quality_drift.png)
