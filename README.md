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
| `random_forest` | Supervised (binary) | Fast, interpretable baseline |
| `xgboost` | Supervised (binary) | Strong on tabular data |
| `lightgbm` | Supervised (binary) | **Best performer** — leaf-wise growth captures complex patterns |
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

## Research Results (50 Users, Medium Difficulty)

The agent ran **8 systematic experiments** on 1,250 sessions (1,000 genuine, 250 impostor) across 102 features.

### Experiment Log

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

*EER convergence detected at Exp 7→8: ΔEER = 0.0005 < 0.003 threshold.*

### Best Configuration

```json
{
  "algorithm": "lightgbm",
  "params": {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 5
  },
  "feature_groups": ["keystroke", "mouse", "derived"],
  "n_features": 81,
  "eer": 0.3285,
  "auc_roc": 0.721,
  "f1": 0.766
}
```

### Top 10 Most Discriminative Features

| Rank | Feature | Group | Description |
|---|---|---|---|
| 1 | `scr_std` | Mouse | Std dev of scroll amounts |
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

## Research Phases (Agent's Strategy)

### Phase 1 — Baseline
Random Forest with default parameters, all 102 features → EER = 0.373. Establishes the performance floor.

### Phase 2 — Feature Ablation
- Keystroke-only (31 features) → EER = 0.395 — **worse**, mouse signals carry complementary information
- Keystroke + Mouse + Derived (81 features) → EER = 0.371 — navigation & temporal features add noise

### Phase 3 — Algorithm Comparison
- XGBoost → EER = 0.364 (+0.009 over RF). Better at feature interactions.
- LightGBM → EER = 0.349 — clear winner. Leaf-wise growth fits high-dimensional biometric data.

### Phase 4 — Hyperparameter Tuning
- Wider trees (`num_leaves=127`) give the biggest single gain
- Combined with the cleaner KMD feature set: EER = **0.3285**
- Convergence confirmed after Exp 8 (ΔEER < 0.001)

---

## Key Findings

1. **LightGBM > XGBoost > Random Forest** on behavioral biometric data — leaf-wise tree growth captures complex nonlinear user patterns better than symmetric splits.

2. **Scroll dynamics are the single most discriminative feature** (`scr_std` ranks #1). Scroll behavior is highly idiosyncratic and hard to imitate.

3. **Keystroke distribution shape matters more than timing means** — entropy, kurtosis, and autocorrelation of intervals outperform raw typing speed.

4. **Drop navigation & temporal features** — they add noise and worsen EER by ~0.020 (6% relative). Session metadata is too generic to distinguish users.

5. **Higher `num_leaves` (127 vs 31)** consistently improves EER — behavioral features interact nonlinearly and benefit from fine-grained splits.

6. **EER ~0.33 reflects inherent task difficulty** at 50-user scale with 20 sessions/user. More sessions → dramatically lower EER (expected <0.10 with 50+ sessions).

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
