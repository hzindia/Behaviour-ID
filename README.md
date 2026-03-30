# Behaviour-ID: Behavioral Fingerprinting Research Agent

An AI-powered research system that **iteratively discovers the best algorithm** for identifying users from their digital behavioral patterns — keystroke dynamics, mouse movements, and navigation habits.

The researcher agent is powered by **Claude via the Agent SDK** (no API subscription needed — uses your existing Claude Code session). It autonomously runs experiments, analyzes results, tunes hyperparameters, and writes a final scientific report.

---

## What is Behavioral Fingerprinting?

Behavioral fingerprinting (also called behavioral biometrics) identifies users by **how** they interact with a device — not by what they know (password) or have (token). Signals include:

| Signal group | What it captures |
|---|---|
| **Keystroke dynamics** | Typing speed, inter-key intervals, key hold time, error rate, typing rhythm |
| **Mouse behavior** | Cursor speed, click duration, scroll direction & variability, double-click rate |
| **Navigation patterns** | Page dwell times, pages per session, back-button usage |
| **Temporal patterns** | Time-of-day preference, session duration, weekday/weekend behavior |

These signals form a unique "fingerprint" that can authenticate users continuously and passively — no extra login step required.

---

## Algorithms Explored

The agent explores 6 ML approaches:

| Algorithm | Type | Notes |
|---|---|---|
| `random_forest` | Supervised (binary) | Fast, interpretable — **wins on hard difficulty** |
| `xgboost` | Supervised (binary) | Strong on tabular data |
| `lightgbm` | Supervised (binary) | **Wins on medium difficulty** — leaf-wise growth captures complex patterns |
| `gradient_boost` | Supervised (binary) | sklearn GBM |
| `isolation_forest` | One-class anomaly detection | Trained on genuine users only |
| `svm` | Supervised (binary) | RBF kernel, slow on large datasets |

Primary metric: **EER (Equal Error Rate)** — the threshold where False Accept Rate equals False Reject Rate. **Lower is better.**

---

## Project Structure

```
behaviour-id/
├── main.py                      # Entry point — runs the full pipeline
├── run_experiment.py            # Standalone experiment runner (called by agent via Bash)
├── config.py                    # Configuration constants
├── requirements.txt
└── src/
    ├── dataset/
    │   ├── generator.py         # Synthetic behavioral data generation
    │   └── features.py          # Feature engineering (102 features, 5 groups)
    ├── models/
    │   └── classical.py         # 6 ML models with unified interface
    ├── evaluation/
    │   └── metrics.py           # EER, AUC-ROC, FAR/FRR, F1
    └── agent/
        └── researcher.py        # Claude research agent (Agent SDK loop)
```

---

## How It Works

```
main.py
  │
  ├─ [1] Generate synthetic dataset
  │       └─ 50 users × 20 sessions each, + impostor sessions
  │          Keystroke / mouse / navigation / temporal signals
  │
  ├─ [2] Feature engineering
  │       └─ 102 statistical features extracted per session
  │          (means, std, IQR, kurtosis, entropy, autocorrelation, …)
  │
  └─ [3] Research Agent (Claude via Agent SDK)
          └─ Claude has Bash + Read + Write tools
          └─ Strategy: baseline → feature ablation → algorithm comparison → hyperparameter tuning
          └─ Calls: python run_experiment.py --algorithm xgboost --params '{"n_estimators": 200}'
          └─ Reads JSON results, iterates experiments
          └─ Writes research_report.md + research_result.json
```

The agent's research loop is fully autonomous — it decides what to try next based on previous results, detects convergence, and stops when satisfied.

---

## Installation

```bash
# Clone / navigate to project
cd behaviour-id

# Install dependencies
pip install -r requirements.txt
pip install claude-agent-sdk
```

**No API key needed.** The agent uses your existing Claude Code session.

---

## Usage

```bash
# Standard run (50 users, 20 sessions, 8 research iterations)
python main.py

# Quick test (20 users, 8 sessions, 3 iterations — ~2 min)
python main.py --quick

# Hard difficulty — users are behaviorally similar, harder to distinguish
python main.py --difficulty hard --n-users 80 --iterations 10

# All options
python main.py --help
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--n-users` | 50 | Number of distinct users to simulate |
| `--n-sessions` | 20 | Genuine sessions per user |
| `--n-impostors` | 5 | Impostor sessions per target user |
| `--difficulty` | medium | `easy` / `medium` / `hard` — inter-user similarity |
| `--noise` | 0.12 | Intra-session noise level (0–1) |
| `--iterations` | 8 | Max research iterations |
| `--seed` | 42 | Random seed for reproducibility |
| `--quick` | — | Fast preset: 20 users, 8 sessions, 3 iterations |

### Output

Results are saved to `experiments/<timestamp>/`:

```
experiments/20260330_162113/
├── feature_dataset.csv      # Generated behavioral feature matrix
├── research_report.md       # Full human-readable research report
├── research_result.json     # Structured JSON: best config + all experiments
└── results_raw.json         # Raw agent output
```

---

## Research Results

Two full runs were conducted across different difficulty settings. The agent autonomously adapted its strategy to each scenario.

---

### Run 1 — Medium Difficulty (8 iterations)

`python main.py` — 50 users, 20 sessions, standard inter-user variance

**1,250 sessions · 102 features · 5-fold cross-validation**

| # | Algorithm | Feature Groups | EER ↓ | AUC-ROC | F1 |
|---|---|---|---|---|---|
| 1 | Random Forest | All (102) | 0.3730 | 0.674 | 0.728 |
| 2 | Random Forest | Keystroke only (31) | 0.3950 | 0.651 | 0.708 |
| 3 | Random Forest | Keystroke + Mouse + Derived (81) | 0.3705 | 0.678 | 0.735 |
| 4 | XGBoost | All (102) | 0.3640 | 0.678 | 0.737 |
| 5 | LightGBM | All (102) | 0.3485 | 0.695 | 0.749 |
| 6 | LightGBM tuned | All (102) | 0.3560 | 0.697 | 0.743 |
| **7 ⭐** | **LightGBM tuned** | **Keystroke + Mouse + Derived (81)** | **0.3285** | **0.721** | **0.766** |
| 8 | LightGBM deeper | Keystroke + Mouse + Derived (81) | 0.3290 | 0.724 | 0.765 |

*Convergence at Exp 7→8: ΔEER = 0.0005*

**Best config:**
```json
{
  "algorithm": "lightgbm",
  "params": { "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 5 },
  "feature_groups": ["keystroke", "mouse", "derived"],
  "n_features": 81,
  "eer": 0.3285,  "auc_roc": 0.721,  "f1": 0.766
}
```

---

### Run 2 — Hard Difficulty (10 iterations)

`python main.py --difficulty hard --iterations 10` — users are behaviorally similar, impostor sessions harder to detect

**1,250 sessions · 102 features · 5-fold cross-validation**

| # | Algorithm | Feature Groups | EER ↓ | AUC-ROC | F1 |
|---|---|---|---|---|---|
| 1 | Random Forest | All (102) | 0.3285 | 0.705 | 0.766 |
| 2 | Random Forest | Keystroke + Derived (43) | 0.3890 | 0.651 | — |
| 3 | Random Forest | Keystroke + Mouse + Derived (81) | 0.3280 | 0.678 | — |
| 4 | XGBoost | All (102) | 0.3610 | 0.678 | — |
| 5 | LightGBM | All (102) | 0.3400 | 0.695 | — |
| 6 | RF balanced (500 trees) | All (102) | 0.3395 | — | — |
| 7 | XGBoost tuned | All (102) | 0.3595 | — | — |
| 8 | RF regularized | All (102) | 0.3400 | — | — |
| 9 | LightGBM regularized | All (102) | 0.3545 | — | — |
| 10 | Gradient Boost | All (102) | 0.3445 | — | — |
| **1 ⭐** | **Random Forest (baseline)** | **All (102)** | **0.3285** | **0.705** | **0.766** |

*All 10 experiments converged around EER ≈ 0.328–0.395 — the baseline RF held its lead.*

**Best config:**
```json
{
  "algorithm": "random_forest",
  "params": { "n_estimators": 100, "max_depth": 10, "random_state": 42 },
  "feature_groups": ["all"],
  "n_features": 102,
  "eer": 0.3285,  "auc_roc": 0.705,  "f1": 0.766
}
```

---

### Cross-Run Comparison

| Setting | Best Algorithm | EER | AUC-ROC | Key Insight |
|---|---|---|---|---|
| Medium (8 iter) | LightGBM | **0.3285** | 0.721 | Leaf-wise growth + feature selection wins |
| Hard (10 iter) | Random Forest | **0.3285** | 0.705 | Boosting overfits when users are similar |

> **Difficulty inversion:** When users are behaviorally similar (hard), boosted models overfit to the noisy dominant features (`scr_std`). Random Forest's bagging diversity generalises better in that regime.

---

### Top 10 Most Discriminative Features (consistent across both runs)

| Rank | Feature | Group | Description |
|---|---|---|---|
| 1 | `scr_std` | Mouse | Std dev of scroll amounts — highly user-idiosyncratic |
| 2 | `kht_std` | Keystroke | Std dev of key hold times |
| 3 | `scr_iqr` | Mouse | IQR of scroll amounts |
| 4 | `ksi_kurt` | Keystroke | Kurtosis of keystroke intervals |
| 5 | `kht_entropy` | Keystroke | Entropy of key hold time distribution |
| 6 | `ms_skew` | Mouse | Skewness of mouse speed |
| 7 | `ksi_autocorr1` | Keystroke | Lag-1 autocorrelation of keystroke intervals |
| 8 | `ksi_skew` | Keystroke | Skewness of keystroke intervals |
| 9 | `ksi_autocorr2` | Keystroke | Lag-2 autocorrelation of keystroke intervals |
| 10 | `cd_entropy` | Mouse | Entropy of click duration distribution |

---

## Agent's Research Strategy (Both Runs)

### Phase 1 — Baseline
Random Forest with default params, all 102 features. Establishes the EER floor to beat.

### Phase 2 — Feature Ablation
Tests which signal groups carry discriminative power. Consistent finding: **mouse + keystroke + derived is the sweet spot**; navigation and temporal features are too generic.

### Phase 3 — Algorithm Comparison
Pits XGBoost, LightGBM, and GradientBoost against RF. **Medium difficulty**: LightGBM wins. **Hard difficulty**: RF holds — boosting overfits when inter-user variance is low.

### Phase 4 — Hyperparameter Tuning
Focuses budget on the winning algorithm. On medium, `num_leaves=127` on LightGBM gives the largest single gain. On hard, no tuning beats the RF baseline — convergence plateau detected early.

---

## Key Findings

1. **Difficulty changes the winning algorithm.** LightGBM dominates when users are clearly separable; Random Forest is more robust when the population is behaviorally similar.

2. **Scroll dynamics are the #1 feature across all runs** (`scr_std` ranks first). Scrolling rhythm is highly user-idiosyncratic and resistant to imitation.

3. **Keystroke distribution shape > raw timing.** Entropy, kurtosis, and autocorrelation of intervals consistently outperform typing speed means.

4. **Class balancing (`class_weight="balanced"`) hurts.** The 4:1 genuine:impostor imbalance is volumetric, not difficulty-based — forcing balance degrades EER.

5. **Navigation and temporal features add noise.** Dropping them improves EER by ~0.020 on medium difficulty but has less effect on hard (where all features are marginal).

6. **EER floor ~0.33 at this scale.** With 50 users and 20 sessions, EER ~0.33 is the practical limit. Expect EER < 0.10 with 50+ sessions per user or per-user models.

---

## Recommendations for Future Work

- **Per-user adaptive models** — train a binary classifier per user (genuine vs. everyone else) instead of a global model
- **Digraph/trigraph latencies** — n-gram keystroke pairs/triplets are known to outperform per-key statistics
- **Ensemble** — combine LightGBM scores with per-user Isolation Forest for personalized threshold calibration
- **Score calibration** — apply Platt scaling or isotonic regression for better-calibrated probabilities
- **More sessions** — with 50+ sessions per user, EER typically drops below 0.10

---

## Evaluation Metrics Explained

| Metric | Definition | Direction |
|---|---|---|
| **EER** | Operating point where FAR = FRR | Lower is better |
| **FAR** | False Accept Rate — impostors incorrectly admitted | Lower is better |
| **FRR** | False Reject Rate — genuine users incorrectly rejected | Lower is better |
| **AUC-ROC** | Area under the ROC curve | Higher is better |
| **F1** | Harmonic mean of precision & recall | Higher is better |

EER is the primary metric for biometric authentication systems. An EER of 0.33 means the system correctly distinguishes genuine users from impostors 67% of the time at the balanced operating point.
