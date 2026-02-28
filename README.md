# Contextual Ad Relevance Scorer

**Matching ads to webpage content using TF-IDF and Cosine Similarity — zero user data required.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Project Demonstrates

This project builds a **contextual advertising engine** that matches ads to webpage content based on semantic similarity — without using any personal user data or cookies.

This is the core ML challenge in contextual advertising: **how do you serve relevant ads using only the content of the page, not who the user is?**

As third-party cookies become obsolete, contextual targeting is emerging as the privacy-first alternative to behavioral targeting. This project explores that problem from first principles.

---

## Results

The model correctly identifies contextually relevant ads across four distinct article domains:

| Article Topic | Top Matched Ad | Relevance Score |
|--------------|----------------|-----------------|
| Electric Vehicles & Battery Tech | Tesla | 0.3643 |
| Healthy Meal Prep | HelloFresh | 0.4405 |
| ML in Finance & Algorithmic Trading | Bloomberg Terminal | 0.5703 |
| Trail Running Guide | Salomon | 0.5074 |

---

## Pipeline

```
Article Content
      │
      ▼
  Preprocessing
  (tokenize, remove stopwords)
      │
      ▼
  TF-IDF Vectorization ◄──── Ad Inventory
  (fit on ad corpus,
   transform article)
      │
      ▼
  Cosine Similarity
  (article vector vs each ad vector)
      │
      ▼
  Ranked Ad Results
  + Explainable Keywords
```

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| **TF-IDF** | Upweights rare, domain-specific terms. 'algorithmic' matters more than 'good' |
| **Cosine Similarity** | Length-invariant — short ads match long articles fairly |
| **From-scratch implementation** | Shows mathematical understanding, not just API calls |
| **Keyword explainability** | Every match has attribution — critical for ad quality review |

---

## Project Structure

```
contextual_ad_relevance/
│
├── ad_relevance_scorer.py          # Core matching engine (TF-IDF + cosine similarity)
├── monitor.py                      # Drift detection & alerting system
├── experiment_tracker.py           # Lightweight experiment tracking (MLflow concept)
├── simulate_production.py          # Full 4-week MLOps lifecycle simulation
├── contextual_ad_relevance.ipynb   # Jupyter notebook with full walkthrough
├── experiment_log.json             # Versioned model run history
├── monitoring_log.json             # Health check history
├── results.json                    # Scoring output
├── report.html                     # HTML visualization
└── README.md
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Jessicalllll/contextual-ad-relevance
cd contextual-ad-relevance

# Install dependencies
pip install pandas

# Run the core scorer
python ad_relevance_scorer.py

# Run the full MLOps simulation (deploy → monitor → drift → retrain)
python simulate_production.py

# Open the HTML report
open report.html

# Or explore the notebook
jupyter notebook contextual_ad_relevance.ipynb
```

---

## How It Works

### 1. Text Preprocessing
Article and ad text are tokenized, lowercased, and cleaned. Stop words (the, and, is...) are removed since they carry no semantic signal.

### 2. TF-IDF Vectorization
Each document is converted to a weighted term vector. Terms that appear frequently in one document but rarely across the corpus get higher weight — making domain-specific vocabulary more important than common words.

$$\text{TF-IDF}(t, d) = \frac{\text{count}(t, d)}{|d|} \times \log\left(\frac{N+1}{df(t)+1}\right) + 1$$

### 3. Cosine Similarity
The angle between the article vector and each ad vector measures semantic overlap. Length-invariant — a short ad and long article can still match well.

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

### 4. Ranking & Explainability
Ads are sorted by relevance score. For each match, the top contributing keywords are surfaced — making the model interpretable and auditable.

---

## MLOps Layer

Beyond the core matching engine, this project includes a full monitoring and experiment tracking system:

### Experiment Tracker (`experiment_tracker.py`)
Lightweight experiment logging that tracks every model run with parameters, metrics, and artifacts — demonstrating the concept behind MLflow:
- Versioned run IDs with timestamps
- Parameter logging (vectorizer type, corpus size, smoothing)
- Metric logging (avg score, top score, vocabulary size)
- Best run selection for deployment decisions

### Drift Detection & Monitoring (`monitor.py`)
Three-dimensional monitoring system that runs on a schedule to catch model degradation:

| Monitor | What It Detects | Alert Threshold |
|---------|----------------|-----------------|
| Score Distribution Drift | Avg relevance dropping from baseline | Warning: 10% drop, Critical: 20% drop |
| Vocabulary Drift (OOV) | New articles using unknown terms | Warning: 15% OOV, Critical: 30% OOV |
| Coverage Monitoring | % articles with valid ad match | Warning: <80%, Critical: <60% |

### Production Simulation (`simulate_production.py`)
Full 4-week MLOps lifecycle demonstration:

```
Week 1: Baseline deployment    → avg score 0.4706  (healthy)
Week 2: Slight topic drift     → avg score 0.3572  (warning)
Week 3: Significant drift      → avg score 0.2605  (critical → retrain triggered)
Week 4: Post-retraining        → avg score 0.4417  (recovered)
```

This mirrors the production MLOps workflow at scale:
- **Airflow** schedules monitoring DAGs on a cadence
- **Alerts** route to PagerDuty/Slack based on severity
- **Retraining** is triggered automatically when thresholds are breached
- **Model versioning** tracks v1.0 → v2.0 with full audit trail

## Evaluation Metrics

- **Top Score**: Model confidence in the best match
- **Avg Top-3 Score**: Overall recommendation quality
- **Discrimination Gap**: Difference between #1 and #3 score — higher = more decisive model

---

## Production Extensions

In a real contextual ad platform at scale, this pipeline would extend to:

1. **Transformer embeddings** (BERT, sentence-transformers) for deeper semantic understanding — synonyms handled naturally
2. **Real-time inference** with sub-100ms latency for RTB auction integration
3. **Distributed processing** with Apache Spark for billions of daily page impressions
4. **A/B testing framework** to compare contextual vs behavioral targeting baselines
5. **MLOps pipeline** with drift monitoring, automated retraining, and model versioning
6. **Brand safety layer** — prevent ads from appearing next to conflicting content

---

## Limitations

| Limitation | Impact |
|-----------|--------|
| TF-IDF misses semantic meaning | 'car' and 'automobile' scored independently |
| No synonym handling | Vocabulary-dependent matching |
| Static ad inventory | No real-time ad updates in this demo |
| No feedback loop | Can't learn from engagement signals |

---
