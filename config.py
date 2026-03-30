"""
Project-wide configuration constants.
Override via CLI flags in main.py or environment variables.
"""

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET = dict(
    n_users              = 50,
    n_sessions_per_user  = 20,
    n_impostors_per_user = 5,
    difficulty           = "medium",   # "easy" | "medium" | "hard"
    noise_level          = 0.12,
    random_seed          = 42,
)

# ─── Research agent ───────────────────────────────────────────────────────────
AGENT = dict(
    model          = "claude-opus-4-6",
    max_iterations = 8,
    cv_folds       = 5,
)

# ─── Performance targets ──────────────────────────────────────────────────────
TARGETS = dict(
    excellent_eer  = 0.05,   # ≤ 5 % EER → excellent biometric system
    acceptable_eer = 0.10,   # ≤ 10 % EER → acceptable
    min_auc        = 0.90,
)

# ─── Feature groups ───────────────────────────────────────────────────────────
DEFAULT_FEATURE_CONFIG = dict(
    use_keystroke   = True,
    use_mouse       = True,
    use_navigation  = True,
    use_temporal    = True,
    use_derived     = True,
)
