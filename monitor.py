"""
Model Monitor — Drift Detection & Alerting
==========================================
Monitors the health of the contextual ad scoring system over time.

In production this would run on a schedule (e.g. Airflow DAG daily)
and alert via PagerDuty or Slack when thresholds are breached.

Three types of monitoring:
1. Score Distribution Drift  — are relevance scores degrading?
2. Vocabulary Drift          — are new articles using unknown terms?
3. Coverage Monitoring       — are ads matching articles at all?

This is the concept behind tools like:
- Weights & Biases (W&B) model monitoring
- Evidently AI for drift detection
- Grafana dashboards for operational metrics
- Astro Observe (what Brendan uses at GumGum) for pipeline observability
"""

import json
import math
import os
from datetime import datetime
from collections import Counter


# ─────────────────────────────────────────────
# Alert System
# ─────────────────────────────────────────────

class Alert:
    """Represents a monitoring alert."""

    SEVERITY_LEVELS = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

    def __init__(self, alert_type: str, severity: str, message: str, value: float,
                 threshold: float):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}[self.severity]
        return (f"{icon} [{self.severity}] {self.alert_type}: {self.message} "
                f"(value={self.value:.4f}, threshold={self.threshold:.4f})")


# ─────────────────────────────────────────────
# Core Monitor
# ─────────────────────────────────────────────

class ModelMonitor:
    """
    Monitors the contextual ad scoring system for degradation and drift.

    Key monitoring dimensions:
    ─────────────────────────
    1. SCORE DISTRIBUTION DRIFT
       Track average relevance scores over time. If scores drop
       significantly, the model may no longer be well-calibrated
       to current article content.

    2. VOCABULARY DRIFT (Out-Of-Vocabulary Rate)
       When new articles use terms the model has never seen,
       those terms get zero weight in TF-IDF — hurting match quality.
       High OOV rate = signal to retrain on fresher corpus.

    3. COVERAGE MONITORING
       What % of articles have at least one ad with score > threshold?
       Low coverage = ads aren't matching content well.

    4. TOP-K SCORE DISTRIBUTION
       Is the gap between rank-1 and rank-3 ads maintaining?
       Shrinking gap = model becoming less decisive = worse UX.
    """

    def __init__(self,
                 score_drop_warning: float = 0.10,
                 score_drop_critical: float = 0.20,
                 oov_warning: float = 0.15,
                 oov_critical: float = 0.30,
                 coverage_warning: float = 0.80,
                 coverage_critical: float = 0.60,
                 min_score_threshold: float = 0.05):
        """
        Initialize monitor with configurable thresholds.

        Args:
            score_drop_warning:  alert if avg score drops by this % from baseline
            score_drop_critical: critical alert if avg score drops by this %
            oov_warning:         alert if >15% of article tokens are unknown
            oov_critical:        critical if >30% unknown
            coverage_warning:    alert if <80% of articles have a strong match
            coverage_critical:   critical if <60% coverage
            min_score_threshold: minimum score to count as a valid match
        """
        self.thresholds = {
            "score_drop_warning": score_drop_warning,
            "score_drop_critical": score_drop_critical,
            "oov_warning": oov_warning,
            "oov_critical": oov_critical,
            "coverage_warning": coverage_warning,
            "coverage_critical": coverage_critical,
            "min_score_threshold": min_score_threshold
        }
        self.baseline = None
        self.monitoring_log = []
        self.alerts = []
        self.log_file = "monitoring_log.json"

    # ─────────────────────────────────────────
    # Baseline
    # ─────────────────────────────────────────

    def set_baseline(self, scoring_results: list, label: str = "initial"):
        """
        Set baseline metrics from initial model deployment.
        Future runs are compared against this baseline.

        This is analogous to establishing your initial model performance
        benchmarks before deploying to production.
        """
        metrics = self._compute_metrics(scoring_results)
        self.baseline = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        print(f"   Baseline set ({label})")
        print(f"   Avg top score:     {metrics['avg_top_score']:.4f}")
        print(f"   Avg top-3 score:   {metrics['avg_top3_score']:.4f}")
        print(f"   Coverage:          {metrics['coverage']:.2%}")
        print(f"   Avg disc. gap:     {metrics['avg_discrimination_gap']:.4f}")

    # ─────────────────────────────────────────
    # Core Metric Computation
    # ─────────────────────────────────────────

    def _compute_metrics(self, scoring_results: list) -> dict:
        """
        Compute core health metrics from a batch of scoring results.

        Args:
            scoring_results: list of dicts with keys:
                             'article_id', 'top_score', 'scores' (list of all scores)
        """
        if not scoring_results:
            return {}

        top_scores = [r["top_score"] for r in scoring_results]
        top3_scores = [r.get("avg_top3_score", r["top_score"]) for r in scoring_results]
        disc_gaps = [r.get("discrimination_gap", 0) for r in scoring_results]

        # Coverage: % of articles with at least one strong match
        min_threshold = self.thresholds["min_score_threshold"]
        covered = sum(1 for r in scoring_results if r["top_score"] >= min_threshold)
        coverage = covered / len(scoring_results)

        return {
            "avg_top_score": sum(top_scores) / len(top_scores),
            "max_top_score": max(top_scores),
            "min_top_score": min(top_scores),
            "avg_top3_score": sum(top3_scores) / len(top3_scores),
            "avg_discrimination_gap": sum(disc_gaps) / len(disc_gaps),
            "coverage": coverage,
            "n_articles": len(scoring_results)
        }

    # ─────────────────────────────────────────
    # Drift Detection
    # ─────────────────────────────────────────

    def check_score_drift(self, current_metrics: dict) -> list:
        """
        Compare current metrics against baseline.
        Generate alerts if degradation exceeds thresholds.
        """
        alerts = []
        if not self.baseline:
            return alerts

        baseline_avg = self.baseline["metrics"]["avg_top_score"]
        current_avg = current_metrics["avg_top_score"]

        if baseline_avg > 0:
            drop_pct = (baseline_avg - current_avg) / baseline_avg

            if drop_pct >= self.thresholds["score_drop_critical"]:
                alerts.append(Alert(
                    alert_type="SCORE_DRIFT",
                    severity="CRITICAL",
                    message=f"Avg relevance score dropped {drop_pct:.1%} from baseline "
                            f"({baseline_avg:.4f} → {current_avg:.4f}). "
                            f"Immediate retraining recommended.",
                    value=drop_pct,
                    threshold=self.thresholds["score_drop_critical"]
                ))
            elif drop_pct >= self.thresholds["score_drop_warning"]:
                alerts.append(Alert(
                    alert_type="SCORE_DRIFT",
                    severity="WARNING",
                    message=f"Avg relevance score dropped {drop_pct:.1%} from baseline. "
                            f"Monitor closely.",
                    value=drop_pct,
                    threshold=self.thresholds["score_drop_warning"]
                ))
            else:
                alerts.append(Alert(
                    alert_type="SCORE_DRIFT",
                    severity="INFO",
                    message=f"Score stable. Change from baseline: {drop_pct:+.1%}",
                    value=drop_pct,
                    threshold=self.thresholds["score_drop_warning"]
                ))

        return alerts

    def check_vocabulary_drift(self, article_tokens: list,
                                known_vocabulary: set) -> list:
        """
        Compute Out-Of-Vocabulary (OOV) rate for new articles.

        High OOV rate means new content is using terms the model
        has never seen — those terms get zero TF-IDF weight,
        hurting match quality silently.

        This is data drift — the input distribution has shifted
        from what the model was trained on.
        """
        alerts = []
        all_tokens = []
        for tokens in article_tokens:
            all_tokens.extend(tokens)

        if not all_tokens:
            return alerts

        oov_tokens = [t for t in all_tokens if t not in known_vocabulary]
        oov_rate = len(oov_tokens) / len(all_tokens)
        oov_unique = set(oov_tokens)

        if oov_rate >= self.thresholds["oov_critical"]:
            alerts.append(Alert(
                alert_type="VOCABULARY_DRIFT",
                severity="CRITICAL",
                message=f"OOV rate {oov_rate:.1%} — {len(oov_unique)} unknown terms. "
                        f"Model vocabulary is stale. Retrain on recent corpus.",
                value=oov_rate,
                threshold=self.thresholds["oov_critical"]
            ))
        elif oov_rate >= self.thresholds["oov_warning"]:
            alerts.append(Alert(
                alert_type="VOCABULARY_DRIFT",
                severity="WARNING",
                message=f"OOV rate {oov_rate:.1%} — new terms include: "
                        f"{list(oov_unique)[:5]}. Consider retraining.",
                value=oov_rate,
                threshold=self.thresholds["oov_warning"]
            ))
        else:
            alerts.append(Alert(
                alert_type="VOCABULARY_DRIFT",
                severity="INFO",
                message=f"Vocabulary healthy. OOV rate: {oov_rate:.1%}",
                value=oov_rate,
                threshold=self.thresholds["oov_warning"]
            ))

        return alerts

    def check_coverage(self, current_metrics: dict) -> list:
        """
        Check what % of articles have at least one meaningful ad match.
        Low coverage = ads aren't relevant to current content.
        """
        alerts = []
        coverage = current_metrics.get("coverage", 1.0)

        if coverage < self.thresholds["coverage_critical"]:
            alerts.append(Alert(
                alert_type="COVERAGE_DROP",
                severity="CRITICAL",
                message=f"Only {coverage:.1%} of articles have a valid ad match. "
                        f"Ad inventory may not cover current content topics.",
                value=coverage,
                threshold=self.thresholds["coverage_critical"]
            ))
        elif coverage < self.thresholds["coverage_warning"]:
            alerts.append(Alert(
                alert_type="COVERAGE_DROP",
                severity="WARNING",
                message=f"Coverage at {coverage:.1%}. "
                        f"Some article topics lack relevant ads.",
                value=coverage,
                threshold=self.thresholds["coverage_warning"]
            ))
        else:
            alerts.append(Alert(
                alert_type="COVERAGE",
                severity="INFO",
                message=f"Coverage healthy at {coverage:.1%}",
                value=coverage,
                threshold=self.thresholds["coverage_warning"]
            ))

        return alerts

    # ─────────────────────────────────────────
    # Run Full Health Check
    # ─────────────────────────────────────────

    def run_health_check(self, scoring_results: list,
                          article_tokens: list,
                          known_vocabulary: set,
                          label: str = "health_check") -> dict:
        """
        Run all monitoring checks and generate a health report.

        In production this would be triggered:
        - On a schedule (daily Airflow DAG)
        - After each batch of new articles is processed
        - When data volume drops unexpectedly

        Args:
            scoring_results: list of scoring result dicts
            article_tokens:  tokenized article content
            known_vocabulary: vocabulary from model training
            label: name for this check (e.g. 'daily_2024_03_15')

        Returns:
            health report dict
        """
        print(f"\nRunning health check: '{label}'")
        print("-" * 50)

        current_metrics = self._compute_metrics(scoring_results)
        all_alerts = []

        # Run all checks
        all_alerts.extend(self.check_score_drift(current_metrics))
        all_alerts.extend(self.check_vocabulary_drift(article_tokens,
                                                       known_vocabulary))
        all_alerts.extend(self.check_coverage(current_metrics))

        # Determine overall health status
        severities = [a.severity for a in all_alerts]
        if "CRITICAL" in severities:
            overall_status = "CRITICAL"
            retraining_recommended = True
        elif "WARNING" in severities:
            overall_status = "WARNING"
            retraining_recommended = False
        else:
            overall_status = "HEALTHY"
            retraining_recommended = False

        # Print alerts
        for alert in all_alerts:
            print(f"  {alert}")

        print(f"\n  Overall Status: {overall_status}")
        if retraining_recommended:
            print("RETRAINING RECOMMENDED")

        # Build report
        report = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.split()[0],
            "metrics": current_metrics,
            "alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "value": round(a.value, 6),
                    "threshold": a.threshold
                }
                for a in all_alerts
            ],
            "retraining_recommended": retraining_recommended
        }

        self.monitoring_log.append(report)
        self.alerts.extend(all_alerts)
        self._save_log()

        return report

    def _save_log(self):
        """Persist monitoring log to disk."""
        with open(self.log_file, 'w') as f:
            json.dump(self.monitoring_log, f, indent=2)

    # ─────────────────────────────────────────
    # Retraining Decision
    # ─────────────────────────────────────────

    def should_retrain(self) -> tuple:
        """
        Evaluate whether model should be retrained based on
        accumulated monitoring history.

        Returns (bool, reason_string)

        In production this would trigger an Airflow DAG or
        a CI/CD pipeline to kick off model retraining.
        """
        if not self.monitoring_log:
            return False, "No monitoring data yet"

        recent = self.monitoring_log[-1]

        if recent.get("retraining_recommended"):
            return True, "Critical alerts detected in latest health check"

        # Check if warning trend is sustained over multiple checks
        if len(self.monitoring_log) >= 3:
            recent_statuses = [r["overall_status"] for r in self.monitoring_log[-3:]]
            if all(s == "WARNING" for s in recent_statuses):
                return True, "Sustained WARNING status over 3 consecutive checks"

        return False, "Model performance within acceptable thresholds"

    def print_monitoring_summary(self):
        """Print a summary of all monitoring runs."""
        if not self.monitoring_log:
            print("No monitoring history yet.")
            return

        print("\nMonitoring History")
        print("=" * 70)
        print(f"{'Label':<25} {'Status':<12} {'Avg Score':<12} {'Coverage':<12} {'Retrain?'}")
        print("-" * 70)
        for report in self.monitoring_log:
            m = report.get("metrics", {})
            print(
                f"{report['label']:<25} "
                f"{report['overall_status']:<12} "
                f"{m.get('avg_top_score', 0):.4f}      "
                f"{m.get('coverage', 0):.2%}       "
                f"{'YES' if report['retraining_recommended'] else 'No'}"
            )
