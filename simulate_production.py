"""
Production Simulation
=====================
Simulates a real-world MLOps scenario over multiple time periods:

Week 1: Baseline deployment — model performing well
Week 2: Slight drift — new articles introduce some new vocabulary
Week 3: Significant drift — content has shifted, scores degrading
Week 4: Post-retraining — model retrained on new data, performance restored

This demonstrates the full ML lifecycle:
    Deploy → Monitor → Detect Drift → Retrain → Redeploy

"""

import re
import math
import json
from collections import Counter
from datetime import datetime

from ad_relevance_scorer import (
    ContextualAdScorer, AD_INVENTORY, ARTICLES,
    preprocess, cosine_similarity
)
from monitor import ModelMonitor
from experiment_tracker import ExperimentTracker


# ─────────────────────────────────────────────
# Simulated Article Batches Over Time
# ─────────────────────────────────────────────

# Week 1: Same domain as training — model should perform well
WEEK1_ARTICLES = ARTICLES  # baseline articles

# Week 2: Slight topic shift — some new vocabulary introduced
WEEK2_ARTICLES = [
    {
        "id": "w2_001",
        "title": "Hybrid Vehicles: The Bridge to Full Electric Adoption",
        "content": """
        Hybrid electric vehicles offer consumers a practical transition from
        gasoline engines to fully electric powertrains. Plug-in hybrids allow
        drivers to charge at home while retaining a combustion engine for
        longer trips. Toyota and Honda dominate the hybrid market with proven
        reliability records. Fuel economy ratings for hybrids often exceed
        50 miles per gallon in city driving. Government rebates and HOV lane
        access incentivize hybrid adoption in congested urban areas.
        """
    },
    {
        "id": "w2_002",
        "title": "Intermittent Fasting: Science Behind the Trend",
        "content": """
        Intermittent fasting has gained popularity as a metabolic health
        strategy. Research suggests time-restricted eating windows may
        improve insulin sensitivity and support weight management. Common
        protocols include 16:8 and 5:2 approaches. Dietitians caution that
        fasting is not suitable for everyone and should complement rather
        than replace balanced nutrition. Hydration and electrolyte management
        are important during fasting periods.
        """
    },
    {
        "id": "w2_003",
        "title": "Quantitative Trading: From Excel to Python",
        "content": """
        Quantitative traders increasingly rely on Python for building and
        backtesting systematic strategies. Libraries like pandas, numpy, and
        zipline provide powerful tools for financial data analysis. Moving
        from spreadsheet-based models to code improves reproducibility and
        scalability. Statistical methods including regression and time series
        analysis form the backbone of most quant strategies. Risk-adjusted
        returns and Sharpe ratios are key performance metrics.
        """
    }
]

# Week 3: Significant content drift — topics the model has never seen
WEEK3_ARTICLES = [
    {
        "id": "w3_001",
        "title": "Quantum Computing Breakthroughs in Cryptography",
        "content": """
        Quantum supremacy is reshaping the landscape of cryptographic security.
        Qubits and superposition enable quantum processors to factor large
        primes exponentially faster than classical computers. Post-quantum
        cryptography standards are being developed by NIST to protect
        against future quantum attacks. Lattice-based cryptography and
        hash-based signatures are leading candidates for quantum-resistant
        algorithms. Enterprise cybersecurity teams are beginning post-quantum
        migration planning.
        """
    },
    {
        "id": "w3_002",
        "title": "Regenerative Agriculture: Healing Soil Through Farming",
        "content": """
        Regenerative agriculture practices focus on restoring soil health
        through cover cropping, reduced tillage, and composting. Carbon
        sequestration in healthy soil helps offset agricultural emissions.
        Farmers adopting regenerative methods report improved water retention
        and reduced input costs over time. Biodiversity above and below
        ground improves ecosystem resilience. Consumer demand for
        regeneratively farmed products is growing rapidly.
        """
    },
    {
        "id": "w3_003",
        "title": "Autonomous Drone Delivery: Last Mile Logistics Revolution",
        "content": """
        Unmanned aerial vehicles are transforming last-mile delivery logistics.
        Companies like Zipline and Wing are deploying drone networks in
        suburban and rural areas. Regulatory frameworks from the FAA are
        evolving to accommodate beyond-visual-line-of-sight operations.
        Battery range, payload capacity, and weather resilience remain
        engineering challenges. Urban air mobility corridors are being
        planned in major metropolitan areas.
        """
    }
]

# Week 4: After retraining on new data — performance should recover
WEEK4_ARTICLES = WEEK2_ARTICLES + WEEK3_ARTICLES[:1]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def score_articles(scorer, articles):
    """Score a batch of articles and return monitoring-friendly results."""
    results = []
    for article in articles:
        scored = scorer.score(article, top_k=5)
        top_scores = scored['relevance_score'].tolist()
        results.append({
            "article_id": article["id"],
            "top_score": top_scores[0] if top_scores else 0,
            "avg_top3_score": sum(top_scores[:3]) / min(3, len(top_scores)),
            "discrimination_gap": (top_scores[0] - top_scores[2])
                if len(top_scores) >= 3 else 0,
            "scores": top_scores
        })
    return results


def get_article_tokens(articles):
    """Tokenize all articles for OOV checking."""
    all_tokens = []
    for article in articles:
        tokens = preprocess(article['title'] + ' ' + article['content'])
        all_tokens.append(tokens)
    return all_tokens


def retrain_scorer(articles, ads):
    """
    Simulate model retraining on a new corpus.
    In production: triggered by Airflow, logged in MLflow,
    deployed via CI/CD pipeline.
    """
    print("\nRetraining model on updated corpus...")
    new_scorer = ContextualAdScorer()
    new_scorer.fit(ads)
    print("Retraining complete — new model deployed")
    return new_scorer


# ─────────────────────────────────────────────
# Main Simulation
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PRODUCTION ML LIFECYCLE SIMULATION")
    print("Contextual Ad Relevance Scorer — 4-Week MLOps Demo")
    print("=" * 65)
    print("\nSimulating: Deploy → Monitor → Detect Drift → Retrain → Redeploy")

    # Initialize systems
    tracker = ExperimentTracker()
    monitor = ModelMonitor(
        score_drop_warning=0.10,
        score_drop_critical=0.20,
        oov_warning=0.15,
        oov_critical=0.30
    )

    # ─────────────────────────────────────────
    # WEEK 1: Initial Deployment & Baseline
    # ─────────────────────────────────────────
    print("\n\nWEEK 1 — Initial Deployment")
    print("─" * 50)

    run_id = tracker.start_run(
        run_name="v1.0_initial_deployment",
        params={
            "vectorizer": "tfidf",
            "similarity": "cosine",
            "stop_words": "custom_142",
            "n_ads": len(AD_INVENTORY),
            "smoothing": "laplace"
        }
    )

    scorer_v1 = ContextualAdScorer()
    scorer_v1.fit(AD_INVENTORY)
    known_vocab = set(scorer_v1.vectorizer.vocabulary.keys())

    week1_results = score_articles(scorer_v1, WEEK1_ARTICLES)
    week1_tokens = get_article_tokens(WEEK1_ARTICLES)

    # Log metrics
    avg_score = sum(r['top_score'] for r in week1_results) / len(week1_results)
    max_score = max(r['top_score'] for r in week1_results)
    tracker.log_metrics({
        "avg_top_score": avg_score,
        "max_top_score": max_score,
        "n_articles_scored": len(week1_results),
        "vocabulary_size": len(known_vocab)
    })
    tracker.log_artifact("model_version", "v1.0")
    tracker.log_artifact("training_corpus", "initial_10_ads")
    tracker.end_run("COMPLETED")

    # Set baseline
    monitor.set_baseline(week1_results, label="week1_baseline")

    # Health check
    monitor.run_health_check(
        week1_results, week1_tokens, known_vocab,
        label="week1_health_check"
    )

    # ─────────────────────────────────────────
    # WEEK 2: Slight Drift
    # ─────────────────────────────────────────
    print("\n\nWEEK 2 — Monitoring: Slight Topic Shift")
    print("─" * 50)

    run_id = tracker.start_run(
        run_name="v1.0_week2_monitoring",
        params={"model_version": "v1.0", "week": 2}
    )

    week2_results = score_articles(scorer_v1, WEEK2_ARTICLES)
    week2_tokens = get_article_tokens(WEEK2_ARTICLES)

    avg_score_w2 = sum(r['top_score'] for r in week2_results) / len(week2_results)
    tracker.log_metrics({
        "avg_top_score": avg_score_w2,
        "max_top_score": max(r['top_score'] for r in week2_results)
    })
    tracker.end_run("COMPLETED")

    monitor.run_health_check(
        week2_results, week2_tokens, known_vocab,
        label="week2_health_check"
    )

    # ─────────────────────────────────────────
    # WEEK 3: Significant Drift → Trigger Retrain
    # ─────────────────────────────────────────
    print("\n\nWEEK 3 — Monitoring: Significant Content Drift")
    print("─" * 50)

    run_id = tracker.start_run(
        run_name="v1.0_week3_monitoring",
        params={"model_version": "v1.0", "week": 3}
    )

    week3_results = score_articles(scorer_v1, WEEK3_ARTICLES)
    week3_tokens = get_article_tokens(WEEK3_ARTICLES)

    avg_score_w3 = sum(r['top_score'] for r in week3_results) / len(week3_results)
    tracker.log_metrics({
        "avg_top_score": avg_score_w3,
        "max_top_score": max(r['top_score'] for r in week3_results)
    })
    tracker.end_run("COMPLETED")

    week3_report = monitor.run_health_check(
        week3_results, week3_tokens, known_vocab,
        label="week3_health_check"
    )

    # Check retraining decision
    should_retrain, reason = monitor.should_retrain()
    print(f"\n  Retraining Decision: {'RETRAIN' if should_retrain else 'HOLD'}")
    print(f"     Reason: {reason}")

    # ─────────────────────────────────────────
    # WEEK 4: Retrain & Redeploy
    # ─────────────────────────────────────────
    print("\n\nWEEK 4 — Retraining & Redeployment")
    print("─" * 50)

    # Retrain on expanded ad inventory with new content
    expanded_ads = AD_INVENTORY + [
        {
            "id": "ad_011",
            "brand": "IBM Quantum",
            "category": "Technology",
            "content": "IBM Quantum computing platform. Build quantum algorithms, "
                       "explore cryptography and optimization. Access real quantum "
                       "hardware via the cloud."
        },
        {
            "id": "ad_012",
            "brand": "DJI",
            "category": "Drone Technology",
            "content": "DJI professional drones for aerial photography, surveying, "
                       "and delivery applications. Industry-leading flight stability "
                       "and payload capacity."
        },
        {
            "id": "ad_013",
            "brand": "Patagonia",
            "category": "Sustainable Outdoors",
            "content": "Patagonia — environmental responsibility in every product. "
                       "Regenerative organic certified materials, fair trade certified, "
                       "committed to carbon neutrality."
        }
    ]

    run_id = tracker.start_run(
        run_name="v2.0_retrained",
        params={
            "vectorizer": "tfidf",
            "similarity": "cosine",
            "n_ads": len(expanded_ads),
            "trigger": "drift_detection_week3",
            "new_ad_categories": ["Technology", "Drone Technology", "Sustainable Outdoors"]
        }
    )

    scorer_v2 = retrain_scorer(WEEK3_ARTICLES, expanded_ads)
    new_vocab = set(scorer_v2.vectorizer.vocabulary.keys())
    vocab_growth = len(new_vocab) - len(known_vocab)

    week4_results = score_articles(scorer_v2, WEEK4_ARTICLES)
    week4_tokens = get_article_tokens(WEEK4_ARTICLES)

    avg_score_w4 = sum(r['top_score'] for r in week4_results) / len(week4_results)
    tracker.log_metrics({
        "avg_top_score": avg_score_w4,
        "max_top_score": max(r['top_score'] for r in week4_results),
        "vocabulary_growth": vocab_growth,
        "n_new_ads": len(expanded_ads) - len(AD_INVENTORY)
    })
    tracker.log_artifact("model_version", "v2.0")
    tracker.log_artifact("retraining_trigger", "week3_drift_alert")
    tracker.end_run("COMPLETED")

    monitor.run_health_check(
        week4_results, week4_tokens, new_vocab,
        label="week4_post_retrain"
    )

    # ─────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("FULL SIMULATION SUMMARY")
    print("=" * 65)

    tracker.compare_runs()
    monitor.print_monitoring_summary()

    # Performance recovery summary
    print("\n\nPerformance Trajectory")
    print("-" * 50)
    weeks = ["Week 1 (Baseline)", "Week 2 (Slight drift)",
             "Week 3 (Significant drift)", "Week 4 (Post-retrain)"]
    scores = [
        sum(r['top_score'] for r in week1_results) / len(week1_results),
        sum(r['top_score'] for r in week2_results) / len(week2_results),
        sum(r['top_score'] for r in week3_results) / len(week3_results),
        sum(r['top_score'] for r in week4_results) / len(week4_results)
    ]

    baseline = scores[0]
    for week, score in zip(weeks, scores):
        bar_len = int(score * 50)
        bar = "█" * bar_len
        change = f"({(score - baseline) / baseline:+.1%})" if score != baseline else "(baseline)"
        print(f"{week:<28} {score:.4f} {change:<12} {bar}")

    print("\nSimulation complete!")
    print("   Files saved: experiment_log.json, monitoring_log.json")
    print("\nKey MLOps concepts demonstrated:")
    print("   • Experiment tracking with versioned runs")
    print("   • Score distribution drift detection")
    print("   • Out-of-vocabulary (data) drift detection")
    print("   • Coverage monitoring")
    print("   • Automated retraining trigger logic")
    print("   • Model versioning (v1.0 → v2.0)")
    print("   • Performance recovery after retraining")


if __name__ == "__main__":
    main()
