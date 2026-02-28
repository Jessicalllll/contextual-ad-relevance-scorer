"""
Experiment Tracker
==================
A lightweight experiment tracking system that logs model runs,
parameters, and metrics — simulating what MLflow does in production.

In production you'd use MLflow or Weights & Biases.
This demonstrates the CONCEPT and WHY those tools exist.

Every model run is logged with:
- Run ID and timestamp
- Model parameters
- Performance metrics
- Artifacts (vocab size, ad count)
"""

import json
import math
import os
from datetime import datetime


TRACKER_FILE = "experiment_log.json"


class ExperimentTracker:
    """
    Lightweight experiment tracker.

    Logs model runs so you can:
    - Compare performance across versions
    - Know exactly which parameters produced which results
    - Roll back to a previous version if a new one degrades
    - Audit model history for debugging

    This is exactly what MLflow's tracking server does at scale.
    """

    def __init__(self, tracker_file: str = TRACKER_FILE):
        self.tracker_file = tracker_file
        self.runs = self._load()
        self.active_run = None

    def _load(self) -> list:
        """Load existing runs from disk."""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return []

    def _save(self):
        """Persist runs to disk."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.runs, f, indent=2)

    def start_run(self, run_name: str, params: dict) -> str:
        """
        Start a new experiment run.

        Args:
            run_name: descriptive name for this run
            params: model hyperparameters and config

        Returns:
            run_id string
        """
        run_id = f"run_{len(self.runs) + 1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_run = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "status": "RUNNING",
            "params": params,
            "metrics": {},
            "artifacts": {}
        }
        print(f"🚀 Started run: {run_id} — '{run_name}'")
        return run_id

    def log_metric(self, key: str, value: float):
        """Log a single metric for the active run."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run["metrics"][key] = round(value, 6)

    def log_metrics(self, metrics: dict):
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, key: str, value):
        """Log an artifact (vocab size, model path, etc.)."""
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run["artifacts"][key] = value

    def end_run(self, status: str = "COMPLETED"):
        """
        End the active run and persist to disk.

        Args:
            status: COMPLETED, FAILED, or CANCELLED
        """
        if not self.active_run:
            raise RuntimeError("No active run to end.")
        self.active_run["status"] = status
        self.active_run["duration_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.active_run["timestamp"])
        ).seconds
        self.runs.append(self.active_run)
        self._save()
        print(f"Run {self.active_run['run_id']} ended — Status: {status}")
        print(f"   Metrics: {self.active_run['metrics']}")
        self.active_run = None

    def get_best_run(self, metric: str, higher_is_better: bool = True) -> dict:
        """
        Find the best run by a given metric.
        This is how you'd select which model version to deploy.
        """
        completed_runs = [r for r in self.runs if r["status"] == "COMPLETED"
                          and metric in r["metrics"]]
        if not completed_runs:
            return None

        return max(completed_runs, key=lambda r: r["metrics"][metric]) \
            if higher_is_better else \
            min(completed_runs, key=lambda r: r["metrics"][metric])

    def compare_runs(self) -> None:
        """Print a comparison table of all completed runs."""
        completed = [r for r in self.runs if r["status"] == "COMPLETED"]
        if not completed:
            print("No completed runs yet.")
            return

        print("\nExperiment Run Comparison")
        print("=" * 80)
        print(f"{'Run ID':<30} {'Name':<25} {'Avg Score':<12} {'Top Score':<12} {'Status'}")
        print("-" * 80)
        for run in completed:
            avg = run["metrics"].get("avg_top_score", "N/A")
            top = run["metrics"].get("max_top_score", "N/A")
            print(f"{run['run_id']:<30} {run['run_name']:<25} {avg:<12} {top:<12} {run['status']}")

    def print_summary(self):
        """Print full summary of all runs."""
        print(f"\n📋 Experiment Log Summary")
        print(f"Total runs: {len(self.runs)}")
        completed = [r for r in self.runs if r['status'] == 'COMPLETED']
        print(f"Completed: {len(completed)}")
        if completed:
            best = self.get_best_run('avg_top_score')
            if best:
                print(f"Best run: {best['run_id']} — avg_top_score: {best['metrics'].get('avg_top_score')}")
