"""
Synthetic behavioral fingerprinting dataset generator.

Generates realistic user behavioral data including:
- Keystroke dynamics (typing speed, key hold time, inter-key intervals)
- Mouse behavior (speed, click duration, scroll patterns)
- Navigation patterns (page dwell times, session structure)
- Temporal patterns (time-of-day preferences)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UserProfile:
    """Encodes a user's unique behavioral fingerprint parameters."""
    user_id: str

    # --- Typing dynamics ---
    mean_typing_speed: float        # characters per second
    typing_speed_std: float
    mean_key_hold_ms: float         # milliseconds
    key_hold_std: float
    error_rate: float               # probability of a typo per keystroke
    burst_typing_prob: float        # probability of a fast-typing "burst"

    # --- Mouse dynamics ---
    mean_mouse_speed: float         # pixels per second
    mouse_speed_std: float
    mean_click_duration_ms: float
    click_duration_std: float
    scroll_direction_bias: float    # −1 (prefers up-scroll) … +1 (prefers down)
    double_click_rate: float        # fraction of clicks that are double-clicks

    # --- Navigation ---
    mean_session_duration: float    # seconds
    session_duration_std: float
    mean_pages_per_session: float
    page_dwell_mean: float          # seconds per page
    page_dwell_std: float
    back_button_rate: float         # fraction of navigations using back

    # --- Temporal ---
    preferred_hour: float           # 0–23
    temporal_spread: float          # std dev in hours


class BehaviorDatasetGenerator:
    """
    Generates a synthetic but statistically realistic dataset for
    behavioral fingerprinting research.

    Design goals
    ~~~~~~~~~~~~
    * Each user has a stable "fingerprint" (profile) that persists
      across sessions with natural intra-session noise.
    * Impostors are drawn from OTHER real user profiles, making the
      problem non-trivial.
    * A ``difficulty`` knob controls how similar users are to each
      other (population-level variance).
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def generate_user_profiles(
        self,
        n_users: int,
        difficulty: str = "medium",
    ) -> List[UserProfile]:
        """
        Sample ``n_users`` distinct behavioral profiles.

        Args:
            n_users:    How many distinct users to model.
            difficulty: Population variance preset.
                        "easy"   → users are very different → low EER expected.
                        "medium" → moderate overlap.
                        "hard"   → users are behaviorally similar → higher EER.
        """
        spread = {"easy": 1.8, "medium": 1.0, "hard": 0.45}.get(difficulty, 1.0)
        rng = self.rng

        profiles: List[UserProfile] = []
        for i in range(n_users):
            p = UserProfile(
                user_id=f"user_{i:04d}",

                # Typing
                mean_typing_speed=np.clip(rng.normal(5.5, 2.2 * spread), 0.8, 16.0),
                typing_speed_std=np.abs(rng.normal(1.2, 0.4)),
                mean_key_hold_ms=np.clip(rng.normal(105, 28 * spread), 40, 250),
                key_hold_std=np.abs(rng.normal(18, 6)),
                error_rate=np.clip(rng.normal(0.035, 0.022 * spread), 0.001, 0.15),
                burst_typing_prob=np.clip(rng.beta(2, 5), 0.0, 0.6),

                # Mouse
                mean_mouse_speed=np.clip(rng.normal(420, 160 * spread), 80, 1100),
                mouse_speed_std=np.abs(rng.normal(110, 35)),
                mean_click_duration_ms=np.clip(rng.normal(115, 38 * spread), 40, 320),
                click_duration_std=np.abs(rng.normal(22, 8)),
                scroll_direction_bias=rng.uniform(-1, 1),
                double_click_rate=np.clip(rng.beta(1.5, 8), 0.0, 0.35),

                # Navigation
                mean_session_duration=np.clip(rng.normal(310, 130 * spread), 30, 1200),
                session_duration_std=np.abs(rng.normal(65, 22)),
                mean_pages_per_session=np.clip(rng.normal(8.5, 3.2 * spread), 1, 40),
                page_dwell_mean=np.clip(rng.normal(38, 16 * spread), 3, 180),
                page_dwell_std=np.abs(rng.normal(12, 4)),
                back_button_rate=np.clip(rng.beta(2, 6), 0.0, 0.5),

                # Temporal
                preferred_hour=rng.uniform(7.5, 22.5),
                temporal_spread=rng.uniform(1.5, 7.0),
            )
            profiles.append(p)

        return profiles

    # ------------------------------------------------------------------
    # Session generation
    # ------------------------------------------------------------------

    def _sample_keystrokes(
        self, profile: UserProfile, n_chars: int, noise: float
    ) -> dict:
        rng = self.rng
        effective_speed = max(0.5, rng.normal(
            profile.mean_typing_speed, profile.typing_speed_std * (1 + noise)
        ))

        intervals, holds = [], []
        for _ in range(max(5, n_chars)):
            # Occasional burst → much shorter interval
            is_burst = rng.random() < profile.burst_typing_prob
            base_interval = 1000.0 / (effective_speed * 5)
            interval = max(10, rng.normal(
                base_interval * (0.4 if is_burst else 1.0),
                base_interval * 0.25 * (1 + noise),
            ))
            hold = max(20, rng.normal(
                profile.mean_key_hold_ms,
                profile.key_hold_std * (1 + noise),
            ))
            intervals.append(interval)
            holds.append(hold)

        return {"intervals": intervals, "holds": holds}

    def _sample_mouse_events(
        self, profile: UserProfile, duration_s: float, noise: float
    ) -> dict:
        rng = self.rng
        n_moves = max(10, int(duration_s * rng.uniform(0.8, 2.5)))
        speeds = [
            max(10, rng.normal(profile.mean_mouse_speed, profile.mouse_speed_std * (1 + noise)))
            for _ in range(n_moves)
        ]

        n_clicks = max(4, int(n_moves * 0.12))
        click_durs = [
            max(10, rng.normal(profile.mean_click_duration_ms, profile.click_duration_std * (1 + noise)))
            for _ in range(n_clicks)
        ]
        is_double = [rng.random() < profile.double_click_rate for _ in range(n_clicks)]

        n_scrolls = max(4, int(n_moves * 0.28))
        scroll_amounts = [
            rng.normal(profile.scroll_direction_bias * 95, 55 * (1 + noise))
            for _ in range(n_scrolls)
        ]

        return {
            "speeds": speeds,
            "click_durations": click_durs,
            "is_double_click": is_double,
            "scroll_amounts": scroll_amounts,
        }

    def generate_session(
        self,
        profile: UserProfile,
        session_idx: int,
        noise_level: float = 0.12,
    ) -> dict:
        """
        Simulate one web-browsing session for a given user profile.

        Returns a dict of raw behavioral signals.
        """
        rng = self.rng

        # ---- Session-level parameters ----
        duration = max(
            20,
            rng.normal(profile.mean_session_duration, profile.session_duration_std * (1 + noise_level)),
        )
        n_pages = max(
            1,
            int(rng.normal(profile.mean_pages_per_session, profile.mean_pages_per_session * 0.3 * (1 + noise_level))),
        )
        n_chars = int(duration * max(0.3, rng.normal(profile.mean_typing_speed, profile.typing_speed_std)))

        # ---- Keystroke events ----
        ks = self._sample_keystrokes(profile, n_chars, noise_level)

        # ---- Mouse events ----
        ms = self._sample_mouse_events(profile, duration, noise_level)

        # ---- Page dwell times ----
        dwells = [
            max(1, rng.normal(profile.page_dwell_mean, profile.page_dwell_std * (1 + noise_level)))
            for _ in range(n_pages)
        ]

        # ---- Temporal context ----
        hour = (profile.preferred_hour + rng.normal(0, profile.temporal_spread)) % 24
        is_weekend = rng.random() < 0.28

        # ---- Back-navigation events ----
        n_back = sum(rng.random() < profile.back_button_rate for _ in range(n_pages))

        return {
            "user_id": profile.user_id,
            "session_id": f"{profile.user_id}_s{session_idx:05d}",
            # Session metadata
            "session_duration": duration,
            "n_pages": n_pages,
            "n_back_nav": n_back,
            "hour_of_day": hour,
            "is_weekend": int(is_weekend),
            # Keystroke signals
            "keystroke_intervals": ks["intervals"],
            "key_hold_times": ks["holds"],
            "n_keystrokes": len(ks["intervals"]),
            "error_rate": np.clip(
                profile.error_rate + rng.normal(0, 0.005 * (1 + noise_level)), 0, 0.3
            ),
            # Mouse signals
            "mouse_speeds": ms["speeds"],
            "click_durations": ms["click_durations"],
            "double_click_count": sum(ms["is_double_click"]),
            "scroll_amounts": ms["scroll_amounts"],
            # Navigation signals
            "page_dwell_times": dwells,
        }

    # ------------------------------------------------------------------
    # Full dataset assembly
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_users: int = 50,
        n_sessions_per_user: int = 20,
        n_impostors_per_user: int = 5,
        difficulty: str = "medium",
        noise_level: float = 0.12,
    ) -> pd.DataFrame:
        """
        Build the complete labelled dataset.

        Columns added beyond the session dict:
            target_user  – which user's account the session claims to be
            is_genuine   – 1 if the session belongs to target_user, else 0
            actual_user  – real user_id (same as target_user for genuine sessions)
        """
        profiles = self.generate_user_profiles(n_users, difficulty)
        profile_map = {p.user_id: p for p in profiles}

        rows: List[dict] = []

        for profile in profiles:
            # Genuine sessions
            for s in range(n_sessions_per_user):
                sess = self.generate_session(profile, s, noise_level)
                sess["target_user"] = profile.user_id
                sess["actual_user"] = profile.user_id
                sess["is_genuine"] = 1
                rows.append(sess)

            # Impostor sessions (other real users attempting access)
            candidates = [p for p in profiles if p.user_id != profile.user_id]
            n_imp = min(n_impostors_per_user, len(candidates))
            impostors = self.rng.choice(candidates, size=n_imp, replace=False)

            for j, imp in enumerate(impostors):
                sess = self.generate_session(imp, n_sessions_per_user + j, noise_level * 1.4)
                sess["target_user"] = profile.user_id
                sess["actual_user"] = imp.user_id
                sess["is_genuine"] = 0
                rows.append(sess)

        return pd.DataFrame(rows)
