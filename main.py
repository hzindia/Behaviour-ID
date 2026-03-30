"""
Behavioral Fingerprinting Research Agent — Entry Point

Powered by the Claude Agent SDK (no API key needed — uses your Claude Code session).

Usage
-----
    python main.py                              # standard run
    python main.py --quick                      # fast 3-iteration test
    python main.py --n-users 80 --difficulty hard --iterations 10

Options
-------
    --n-users N         Distinct users to simulate              [default: 50]
    --n-sessions N      Genuine sessions per user               [default: 20]
    --n-impostors N     Impostor sessions per target user       [default: 5]
    --difficulty STR    "easy" | "medium" | "hard"             [default: medium]
    --noise F           Intra-session noise level (0–1)        [default: 0.12]
    --iterations N      Max research iterations                 [default: 8]
    --seed N            Random seed                             [default: 42]
    --quick             Fast preset: 20 users, 8 sessions, 3 iterations
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Behavioral Fingerprinting Research Agent (Agent SDK)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--n-users",     type=int,   default=50)
    p.add_argument("--n-sessions",  type=int,   default=20)
    p.add_argument("--n-impostors", type=int,   default=5)
    p.add_argument("--difficulty",  type=str,   default="medium",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--noise",       type=float, default=0.12)
    p.add_argument("--iterations",  type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--quick",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.n_users     = 20
        args.n_sessions  = 8
        args.n_impostors = 3
        args.iterations  = 3
        print("[quick mode] Reduced dataset for fast testing.")

    # ── Output directory ──────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate synthetic behavioral dataset ─────────────────
    print("\n[1/3] Generating synthetic behavioral dataset...")
    from src.dataset.generator import BehaviorDatasetGenerator

    gen    = BehaviorDatasetGenerator(random_seed=args.seed)
    raw_df = gen.generate_dataset(
        n_users              = args.n_users,
        n_sessions_per_user  = args.n_sessions,
        n_impostors_per_user = args.n_impostors,
        difficulty           = args.difficulty,
        noise_level          = args.noise,
    )
    print(
        f"    {len(raw_df)} sessions  "
        f"({(raw_df['is_genuine']==1).sum()} genuine, "
        f"{(raw_df['is_genuine']==0).sum()} impostor)"
    )

    # ── Step 2: Feature engineering ───────────────────────────────────
    print("\n[2/3] Extracting behavioral features...")
    from src.dataset.features import build_feature_matrix

    feat_df    = build_feature_matrix(raw_df)
    meta_cols  = ["user_id", "target_user", "actual_user", "is_genuine", "session_id"]
    n_features = len([c for c in feat_df.columns if c not in meta_cols])
    print(f"    {len(feat_df)} sessions × {n_features} features")

    # ── Step 3: Research agent ────────────────────────────────────────
    print("\n[3/3] Launching Research Agent (Claude via Agent SDK)...")
    from src.agent.researcher import BehaviorResearchAgent

    agent   = BehaviorResearchAgent(output_dir=output_dir)
    results = agent.run(
        feature_df     = feat_df,
        max_iterations = args.iterations,
        verbose        = True,
    )

    # ── Save raw results ──────────────────────────────────────────────
    def _serial(obj):
        if isinstance(obj, dict):   return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [_serial(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_dir / "results_raw.json", "w") as f:
        json.dump(_serial(results), f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────
    bm = results.get("best_metrics", {})
    bc_algo   = results.get("best_algorithm", "–")
    bc_params = results.get("best_params", {})

    print(f"\n{'═'*60}")
    print("  RESEARCH COMPLETE")
    print(f"{'═'*60}")
    print(f"  Algorithm  : {bc_algo}")
    print(f"  Params     : {bc_params}")
    if bm:
        print(f"  EER        : {bm.get('eer', '–'):.4f}  (lower is better)")
        print(f"  AUC-ROC    : {bm.get('auc_roc', '–'):.4f}")
        print(f"  F1         : {bm.get('f1', '–'):.4f}")
        print(f"  Accuracy   : {bm.get('accuracy', '–'):.4f}")
    print(f"\n  Output dir : {output_dir}/")
    print(f"  Report     : {output_dir}/research_report.md")
    print(f"{'═'*60}\n")

    return results


if __name__ == "__main__":
    main()
