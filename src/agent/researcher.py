"""
Behavioral Fingerprinting Research Agent — Agent SDK version.

Uses the Claude Agent SDK (claude_agent_sdk) which runs through the
Claude Code CLI — no separate Anthropic API key or subscription needed.

How it works
~~~~~~~~~~~~
1. The dataset is saved to disk.
2. An Agent SDK session is launched giving Claude access to Bash, Read, Write.
3. Claude drives the research loop by calling:
       python run_experiment.py --algorithm ... --params '...' ...
   and reading the JSON output.
4. After iterating, Claude writes a final report to disk.
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import anyio
import pandas as pd

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    TextBlock,
    query,
)


class BehaviorResearchAgent:
    """
    AI researcher agent powered by Claude via the Agent SDK.
    No API key required — uses your existing Claude Code session.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_result: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        feature_df: pd.DataFrame,
        max_iterations: int = 8,
        verbose: bool = True,
    ) -> dict:
        """
        Save the feature matrix and launch the agentic research loop.

        Returns a dict with best_metrics, best_config, and final_analysis.
        """
        dataset_path = self.output_dir / "feature_dataset.csv"
        feature_df.to_csv(dataset_path, index=False)

        meta_cols = ["user_id", "target_user", "actual_user", "is_genuine", "session_id"]
        n_features = len([c for c in feature_df.columns if c not in meta_cols])
        n_genuine  = int((feature_df["is_genuine"] == 1).sum())
        n_impostor = int((feature_df["is_genuine"] == 0).sum())
        n_users    = feature_df["target_user"].nunique()

        prompt = self._build_prompt(
            dataset_path   = str(dataset_path),
            output_dir     = str(self.output_dir),
            n_users        = n_users,
            n_sessions     = len(feature_df),
            n_genuine      = n_genuine,
            n_impostor     = n_impostor,
            n_features     = n_features,
            max_iterations = max_iterations,
        )

        if verbose:
            print(f"\n{'═'*68}")
            print("  BEHAVIORAL FINGERPRINTING RESEARCH AGENT")
            print(f"{'═'*68}")
            print(f"  Users      : {n_users}")
            print(f"  Sessions   : {len(feature_df)}  ({n_genuine} genuine, {n_impostor} impostor)")
            print(f"  Features   : {n_features}")
            print(f"  Iterations : {max_iterations}")
            print(f"  Output dir : {self.output_dir}")
            print(f"{'═'*68}\n")
            print("Launching Claude research agent (via Agent SDK)...\n")

        result = anyio.run(self._run_async, prompt, verbose)
        return result

    # ------------------------------------------------------------------
    # Async runner
    # ------------------------------------------------------------------

    async def _run_async(self, prompt: str, verbose: bool) -> dict:
        agent_output_parts = []

        options = ClaudeAgentOptions(
            allowed_tools   = ["Bash", "Read", "Write"],
            permission_mode = "acceptEdits",
            cwd             = str(Path(__file__).resolve().parents[2]),  # project root
            system_prompt   = self._system_prompt(),
            max_turns       = 500,
        )

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                agent_output_parts.append(message.result or "")
                if verbose and message.result:
                    print(f"\n[Agent Result]\n{message.result}")
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and verbose:
                        # Print short snippets so the user sees progress
                        snippet = block.text.strip()
                        if snippet:
                            print(f"\n  {snippet[:400]}")

        # Try to load the structured report written by the agent
        report_json = self.output_dir / "research_result.json"
        if report_json.exists():
            with open(report_json) as f:
                self.final_result = json.load(f)
        else:
            self.final_result = {"raw_output": "\n".join(agent_output_parts)}

        return self.final_result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return textwrap.dedent("""
            You are a world-class ML researcher specialising in behavioral biometrics
            and continuous user authentication.

            Research principles
            -------------------
            • Think scientifically: form hypotheses, test them, draw conclusions.
            • Optimise EER (Equal Error Rate) — LOWER is better.
              EER < 0.10 is good; EER < 0.05 is excellent.
            • Run experiments via Bash using run_experiment.py (JSON output).
            • Parse JSON results carefully; track every experiment's metrics.
            • Explore systematically: baseline → feature ablation → hyperparameter tuning.
            • Keystroke dynamics are usually the most discriminative signal.
            • Tree ensembles (XGBoost, LightGBM, RF) outperform SVMs on tabular data.
            • Stop when EER has converged (< 0.003 improvement over 3 trials) or you've
              run the allowed number of experiments.

            CRITICAL execution rules
            ------------------------
            • NEVER run commands with & (background). Every experiment must run
              synchronously: python run_experiment.py ... (no & at the end).
            • NEVER use parallel/concurrent bash commands. Run ONE experiment at a time,
              wait for it to finish, read the result, then decide the next step.
            • Running background processes WILL crash the session. This is forbidden.
            • Do not use pkill, kill, or any process management commands.

            Output protocol
            ---------------
            • Print a short summary after EVERY experiment (algorithm, EER, AUC).
            • After all experiments, write your final structured report to
              {output_dir}/research_report.md  AND  {output_dir}/research_result.json
            • The JSON must have keys: best_algorithm, best_params, best_metrics,
              feature_groups_used, analysis, all_experiments (list).
        """).strip()

    def _build_prompt(
        self,
        dataset_path: str,
        output_dir: str,
        n_users: int,
        n_sessions: int,
        n_genuine: int,
        n_impostor: int,
        n_features: int,
        max_iterations: int,
    ) -> str:
        return textwrap.dedent(f"""
            # Behavioral Fingerprinting Research Task

            ## Dataset
            - Path: `{dataset_path}`
            - Users: {n_users}
            - Sessions: {n_sessions} ({n_genuine} genuine, {n_impostor} impostor)
            - Pre-engineered features: {n_features}
            - Label column: `is_genuine` (1 = genuine user, 0 = impostor)

            ## Experiment runner
            Use this command to run an experiment (always synchronous, never use &):

            ```bash
            python run_experiment.py \\
                --dataset "{dataset_path}" \\
                --algorithm ALGORITHM \\
                --params '{{"key": value}}' \\
                --feature-groups keystroke,mouse,navigation,temporal,derived \\
                --cv-folds 5 \\
                --label "short_label"
            ```

            IMPORTANT: Run ONE experiment at a time. Wait for JSON output. Never use & or
            background processes. Never run multiple experiments simultaneously.

            Available algorithms:
            - `random_forest`    — RF classifier (fast, good baseline)
            - `xgboost`          — XGBoost gradient boosting (often best)
            - `lightgbm`         — LightGBM (fast, excellent on tabular)
            - `gradient_boost`   — sklearn GBM
            - `isolation_forest` — one-class anomaly detection
            - `svm`              — SVM with RBF kernel (slow on large datasets)

            Feature groups (mix-and-match with comma-separated list):
            - `keystroke`   — typing speed, intervals, hold times, error rate
            - `mouse`       — cursor speed, click duration, scroll direction
            - `navigation`  — page dwell times, pages/session, back-button rate
            - `temporal`    — hour of day, session duration, weekend flag
            - `derived`     — compound ratios (chars/sec, mouse/click ratio, etc.)

            Key hyperparameters to tune:
            - RF/XGB/LGB: `n_estimators` (50–500), `max_depth` (3–15), `learning_rate` (0.01–0.2)
            - XGB extra: `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`
            - LGB extra: `num_leaves` (15–127), `min_child_samples`
            - Isolation Forest: `contamination` (0.05–0.25), `max_features` (0.5–1.0)

            ## Your goal
            Run up to **{max_iterations} experiments** to find the configuration with
            the lowest EER. Be systematic:
            1. Baseline (random_forest, defaults, all features)
            2. Feature ablation (which groups matter most?)
            3. Algorithm comparison (try xgboost, lightgbm)
            4. Hyperparameter tuning on best algorithm
            5. Final validation

            ## Output
            After all experiments write:
            1. `{output_dir}/research_report.md` — human-readable report
            2. `{output_dir}/research_result.json` — structured JSON:
               {{
                 "best_algorithm": "...",
                 "best_params": {{}},
                 "best_metrics": {{"eer": ..., "auc_roc": ..., "f1": ...}},
                 "feature_groups_used": [...],
                 "analysis": "detailed explanation",
                 "all_experiments": [...]
               }}

            Start now. Begin with the baseline experiment.
        """).strip()
