"""
Feature engineering for behavioral fingerprinting.

Transforms raw session signals into a tabular feature vector
suitable for ML classification.

Feature groups
~~~~~~~~~~~~~~
* ksi_*    – keystroke interval statistics
* kht_*    – key hold-time statistics
* ms_*     – mouse speed statistics
* cd_*     – click duration statistics
* scr_*    – scroll-amount statistics
* dwell_*  – page dwell-time statistics
* nav_*    – navigation-level features
* temp_*   – temporal / session-level features
* derived_*– ratio / compound features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# Low-level statistical helpers
# ---------------------------------------------------------------------------

def _stat_features(values: list, prefix: str) -> Dict[str, float]:
    """
    Compute a rich set of descriptive statistics for a 1-D signal.

    Returns a flat dict with keys ``{prefix}_{stat}``.
    Falls back to zeros when fewer than 2 observations are present.
    """
    if not values or len(values) < 2:
        zero = {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
            f"{prefix}_entropy": 0.0,
        }
        return zero

    arr = np.array(values, dtype=np.float64)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])

    # Approximate distribution entropy via histogram
    counts, _ = np.histogram(arr, bins=min(10, len(arr)))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # Skewness / kurtosis (pandas handles edge-cases nicely)
    s = pd.Series(arr)

    return {
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(std),
        f"{prefix}_cv": float(std / (abs(mean) + 1e-9)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_range": float(arr.max() - arr.min()),
        f"{prefix}_skew": float(s.skew()),
        f"{prefix}_kurt": float(s.kurt()),
        f"{prefix}_entropy": entropy,
    }


def _rhythm_features(intervals: list, prefix: str) -> Dict[str, float]:
    """
    Temporal regularity features for interval sequences.

    Captures rhythm / cadence beyond simple statistics.
    """
    if not intervals or len(intervals) < 3:
        return {
            f"{prefix}_autocorr1": 0.0,
            f"{prefix}_autocorr2": 0.0,
            f"{prefix}_jitter": 0.0,
        }
    arr = np.array(intervals, dtype=np.float64)
    n = len(arr)

    # Lag-1 and lag-2 autocorrelation
    def _ac(a, lag):
        if n <= lag:
            return 0.0
        x, y = a[:-lag], a[lag:]
        std_xy = x.std() * y.std()
        if std_xy < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # Jitter: mean absolute difference between consecutive intervals
    jitter = float(np.abs(np.diff(arr)).mean())

    return {
        f"{prefix}_autocorr1": _ac(arr, 1),
        f"{prefix}_autocorr2": _ac(arr, 2),
        f"{prefix}_jitter": jitter,
    }


# ---------------------------------------------------------------------------
# Session-level feature extractor
# ---------------------------------------------------------------------------

def extract_session_features(session: dict) -> Dict[str, float]:
    """
    Convert one raw session dict → flat feature vector (dict).

    This is the default feature extractor; you can swap it out by
    passing a custom callable to ``build_feature_matrix``.
    """
    feat: Dict[str, float] = {}

    # ---- Keystroke dynamics ----
    ksi = session.get("keystroke_intervals", [])
    kht = session.get("key_hold_times", [])
    feat.update(_stat_features(ksi, "ksi"))
    feat.update(_rhythm_features(ksi, "ksi"))
    feat.update(_stat_features(kht, "kht"))
    feat["error_rate"] = float(session.get("error_rate", 0.0))
    feat["n_keystrokes"] = float(session.get("n_keystrokes", len(ksi)))

    # ---- Mouse dynamics ----
    speeds = session.get("mouse_speeds", [])
    cdurs = session.get("click_durations", [])
    scrolls = session.get("scroll_amounts", [])
    feat.update(_stat_features(speeds, "ms"))
    feat.update(_stat_features(cdurs, "cd"))
    feat.update(_stat_features(scrolls, "scr"))

    if scrolls:
        arr = np.array(scrolls)
        feat["scr_pos_ratio"] = float((arr > 0).mean())
        feat["scr_neg_ratio"] = float((arr < 0).mean())
    else:
        feat["scr_pos_ratio"] = 0.5
        feat["scr_neg_ratio"] = 0.5

    feat["double_click_count"] = float(session.get("double_click_count", 0))
    n_clicks = max(1, len(cdurs))
    feat["double_click_rate"] = feat["double_click_count"] / n_clicks

    # ---- Navigation ----
    dwells = session.get("page_dwell_times", [])
    feat.update(_stat_features(dwells, "dwell"))
    feat["n_pages"] = float(session.get("n_pages", 1))
    feat["n_back_nav"] = float(session.get("n_back_nav", 0))
    feat["back_nav_rate"] = feat["n_back_nav"] / max(1, feat["n_pages"])

    # ---- Session / temporal ----
    dur = float(session.get("session_duration", 1.0))
    hour = float(session.get("hour_of_day", 12.0))
    feat["session_duration"] = dur
    feat["is_weekend"] = float(session.get("is_weekend", 0))
    # Encode hour cyclically so 23→0 wraps correctly
    feat["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    feat["hour_raw"] = hour

    # ---- Derived compound features ----
    n_ks = max(1, len(ksi))
    feat["derived_chars_per_sec"] = n_ks / max(1, dur)
    feat["derived_clicks_per_sec"] = n_clicks / max(1, dur)
    feat["derived_pages_per_min"] = feat["n_pages"] / max(1, dur / 60)
    feat["derived_ms_cd_ratio"] = feat["ms_mean"] / max(1, feat["cd_mean"])
    feat["derived_ksi_kht_ratio"] = feat["ksi_mean"] / max(1, feat["kht_mean"])
    # Typing regularity index: lower CV → more regular (higher score)
    feat["derived_typing_regularity"] = 1.0 / (1.0 + feat["ksi_cv"])
    # Mouse smoothness: lower CV → smoother
    feat["derived_mouse_smoothness"] = 1.0 / (1.0 + feat["ms_cv"])

    return feat


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    raw_df: pd.DataFrame,
    feature_extractor: Optional[Callable[[dict], Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Apply ``feature_extractor`` to every row and return a feature DataFrame.

    The metadata columns (user_id, target_user, is_genuine, session_id,
    actual_user) are preserved alongside the features.
    """
    if feature_extractor is None:
        feature_extractor = extract_session_features

    meta_cols = ["user_id", "target_user", "actual_user", "is_genuine", "session_id"]

    rows: List[dict] = []
    for _, row in raw_df.iterrows():
        feat = feature_extractor(row.to_dict())
        for col in meta_cols:
            if col in row.index:
                feat[col] = row[col]
        rows.append(feat)

    feat_df = pd.DataFrame(rows)

    # Move metadata to front for clarity
    existing_meta = [c for c in meta_cols if c in feat_df.columns]
    other = [c for c in feat_df.columns if c not in meta_cols]
    feat_df = feat_df[existing_meta + other]

    return feat_df


# ---------------------------------------------------------------------------
# Feature-group filter utility
# ---------------------------------------------------------------------------

FEATURE_GROUP_PREFIXES = {
    "keystroke": ("ksi_", "kht_", "error_rate", "n_keystrokes"),
    "mouse":     ("ms_", "cd_", "scr_", "double_click"),
    "navigation": ("dwell_", "n_pages", "n_back", "back_nav"),
    "temporal":  ("session_duration", "hour_", "is_weekend"),
    "derived":   ("derived_",),
}


def filter_features(
    feat_df: pd.DataFrame,
    feature_config: dict,
) -> pd.DataFrame:
    """
    Return a copy of ``feat_df`` with only the columns allowed by
    ``feature_config``.

    ``feature_config`` keys:
        use_keystroke, use_mouse, use_navigation, use_temporal, use_derived
        (all default True)
    """
    meta_cols = ["user_id", "target_user", "actual_user", "is_genuine", "session_id"]
    keep = list(meta_cols)

    for group, prefixes in FEATURE_GROUP_PREFIXES.items():
        flag = feature_config.get(f"use_{group}", True)
        if not flag:
            continue
        for col in feat_df.columns:
            if col in meta_cols:
                continue
            if any(col.startswith(pfx) or col == pfx.rstrip("_") for pfx in prefixes):
                if col not in keep:
                    keep.append(col)

    # Safety: always include everything if nothing matched
    if len(keep) == len(meta_cols):
        keep = list(feat_df.columns)

    return feat_df[[c for c in keep if c in feat_df.columns]]
