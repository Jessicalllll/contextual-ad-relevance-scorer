"""
Contextual Ad Relevance Scorer
================================
Author: Jessica Guo
GitHub: https://github.com/Jessicalllll

This project demonstrates contextual advertising — matching ads to webpage
content based on semantic similarity, without using any personal user data.

This directly mirrors the core ML challenge at companies like GumGum:
how do you serve relevant ads using only the content of the page,
not who the user is?

Pipeline:
    1. Parse and preprocess article content
    2. Extract TF-IDF feature vectors for articles and ads
    3. Compute cosine similarity between article and each ad
    4. Rank and return most contextually relevant ads
    5. Visualize results
"""

import re
import math
import pandas as pd
import json
from collections import Counter


# ─────────────────────────────────────────────
# Sample Data — Articles and Ad Inventory
# ─────────────────────────────────────────────

ARTICLES = [
    {
        "id": "article_001",
        "title": "The Future of Electric Vehicles: Battery Technology Breakthroughs",
        "content": """
        Electric vehicles are rapidly transforming the automotive industry.
        Advances in lithium-ion battery technology are pushing range beyond 400 miles
        on a single charge. Major automakers including Tesla, Ford, and GM are investing
        billions in EV infrastructure and manufacturing. Charging networks are expanding
        across highways and urban centers. Government incentives and tax credits are
        making EVs more accessible to mainstream consumers. The environmental benefits
        of zero-emission vehicles are driving adoption among eco-conscious buyers.
        Battery recycling programs are addressing sustainability concerns about
        end-of-life battery disposal. Solid-state batteries promise even greater
        energy density and faster charging times in the coming years.
        """
    },
    {
        "id": "article_002",
        "title": "Healthy Meal Prep: How to Eat Well on a Budget",
        "content": """
        Meal prepping is one of the most effective strategies for maintaining a
        healthy diet while saving money. Planning your weekly meals in advance reduces
        food waste and eliminates the temptation of expensive takeout. Nutritionists
        recommend focusing on whole foods like vegetables, lean proteins, and whole grains.
        Batch cooking staples like rice, quinoa, and roasted vegetables saves time
        throughout the week. Investing in quality food storage containers keeps
        meals fresh longer. Simple recipes with minimal ingredients make the process
        sustainable long-term. Tracking macros and calories helps achieve fitness goals
        while staying within budget constraints.
        """
    },
    {
        "id": "article_003",
        "title": "Machine Learning in Finance: Algorithmic Trading Strategies",
        "content": """
        Machine learning is revolutionizing financial markets through sophisticated
        algorithmic trading systems. Hedge funds and investment banks deploy neural
        networks to identify patterns in market data and execute trades at millisecond
        speeds. Natural language processing analyzes news sentiment and earnings calls
        to predict stock movements. Risk management models use historical data to
        optimize portfolio allocation and minimize drawdown. Regulatory frameworks
        are evolving to address the challenges of AI-driven trading. Retail investors
        are gaining access to algorithmic tools through fintech platforms.
        Backtesting frameworks allow strategies to be validated against historical
        market conditions before live deployment.
        """
    },
    {
        "id": "article_004",
        "title": "Trail Running: A Beginner's Guide to Getting Started",
        "content": """
        Trail running offers a refreshing alternative to road running with scenic
        routes through forests, mountains, and national parks. Beginners should
        invest in proper trail running shoes with good grip and ankle support.
        Building a base fitness level through regular running before tackling
        technical terrain is essential. Hydration packs and nutrition gels are
        important for longer trail runs. Learning to read trail maps and using
        GPS watches improves navigation on unmarked paths. Running clubs and
        local trail communities provide support and motivation for new runners.
        Recovery nutrition with protein and carbohydrates aids muscle repair
        after challenging workouts on uneven terrain.
        """
    }
]

AD_INVENTORY = [
    {
        "id": "ad_001",
        "brand": "Tesla",
        "category": "Automotive",
        "content": "Experience the future of driving. Tesla Model 3 — 358 mile range, "
                   "autopilot, zero emissions. Order yours today with federal tax credits available."
    },
    {
        "id": "ad_002",
        "brand": "ChargePoint",
        "category": "EV Infrastructure",
        "content": "ChargePoint home charging station. Charge your electric vehicle overnight. "
                   "Fast, reliable, smart energy management for EV owners."
    },
    {
        "id": "ad_003",
        "brand": "HelloFresh",
        "category": "Food & Nutrition",
        "content": "Fresh ingredients, simple recipes, delivered to your door. "
                   "HelloFresh meal kits make healthy eating easy and affordable. First box 60% off."
    },
    {
        "id": "ad_004",
        "brand": "MyFitnessPal",
        "category": "Health & Fitness",
        "content": "Track calories, macros, and nutrition with MyFitnessPal. "
                   "Over 14 million foods in our database. Reach your health goals faster."
    },
    {
        "id": "ad_005",
        "brand": "Robinhood",
        "category": "Finance",
        "content": "Commission-free stock trading. Invest in stocks, ETFs, and crypto "
                   "with Robinhood. Advanced charts, real-time data, and portfolio analytics."
    },
    {
        "id": "ad_006",
        "brand": "Bloomberg Terminal",
        "category": "Finance",
        "content": "Bloomberg Terminal — real-time financial data, news analytics, "
                   "and algorithmic trading tools trusted by top hedge funds and investment banks."
    },
    {
        "id": "ad_007",
        "brand": "Salomon",
        "category": "Outdoor Sports",
        "content": "Salomon trail running shoes. Superior grip, lightweight design, "
                   "and ankle support for technical terrain. Built for runners who push limits."
    },
    {
        "id": "ad_008",
        "brand": "Garmin",
        "category": "Fitness Technology",
        "content": "Garmin Forerunner GPS watch. Track pace, elevation, heart rate, "
                   "and route navigation for trail and road running. Built for endurance athletes."
    },
    {
        "id": "ad_009",
        "brand": "GE Appliances",
        "category": "Home",
        "content": "GE smart refrigerators with WiFi connectivity and energy efficiency. "
                   "Keep your family's food fresh longer with advanced cooling technology."
    },
    {
        "id": "ad_010",
        "brand": "Coursera",
        "category": "Education",
        "content": "Learn machine learning and data science online with Coursera. "
                   "Courses from Stanford, Google, and top universities. Advance your tech career."
    }
]


# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'their', 'our', 'your', 'my',
    'we', 'you', 'he', 'she', 'his', 'her', 'as', 'more', 'most', 'into',
    'through', 'about', 'than', 'also', 'even', 'such', 'while', 'before',
    'after', 'through', 'across', 'between', 'among', 'how', 'what', 'which'
}


def preprocess(text: str) -> list[str]:
    """
    Clean and tokenize text.
    - Lowercase
    - Remove punctuation
    - Remove stop words
    - Remove short tokens
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens


# ─────────────────────────────────────────────
# TF-IDF Vectorizer
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    """
    Custom TF-IDF implementation from scratch.

    TF-IDF (Term Frequency - Inverse Document Frequency) measures how
    important a word is to a document relative to a corpus.

    - TF: how often a term appears in a document
    - IDF: how rare a term is across all documents (rare = more informative)
    - TF-IDF = TF * IDF

    Why TF-IDF for contextual advertising?
    Common words like 'the' or 'good' get downweighted.
    Domain-specific words like 'lithium-ion' or 'algorithmic' get upweighted.
    This gives us a meaningful semantic fingerprint of each document.
    """

    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.num_docs = 0

    def fit(self, documents: list[list[str]]):
        """Learn vocabulary and IDF scores from corpus."""
        self.num_docs = len(documents)

        # Build vocabulary
        all_terms = set()
        for doc in documents:
            all_terms.update(doc)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Compute IDF for each term
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freq[term] += 1

        for term, df in doc_freq.items():
            # Smoothed IDF: log((N + 1) / (df + 1)) + 1
            self.idf_scores[term] = math.log((self.num_docs + 1) / (df + 1)) + 1

        return self

    def transform(self, document: list[str]) -> dict[str, float]:
        """
        Convert a tokenized document into a TF-IDF vector.
        Returns a sparse dict representation {term: tfidf_score}.
        """
        tf = Counter(document)
        total_terms = len(document)
        tfidf_vector = {}

        for term, count in tf.items():
            if term in self.vocabulary:
                term_freq = count / total_terms
                idf = self.idf_scores.get(term, 0)
                tfidf_vector[term] = term_freq * idf

        return tfidf_vector

    def fit_transform(self, documents: list[list[str]]) -> list[dict[str, float]]:
        """Fit on corpus and transform all documents."""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


# ─────────────────────────────────────────────
# Cosine Similarity
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Compute cosine similarity between two sparse TF-IDF vectors.

    Cosine similarity measures the angle between two vectors.
    Score of 1.0 = identical direction (highly relevant)
    Score of 0.0 = orthogonal (no shared context)

    Why cosine similarity for ad matching?
    It's length-invariant — a short ad and a long article can still
    match well if they share the same key terms, regardless of length.
    """
    # Find common terms
    common_terms = set(vec_a.keys()) & set(vec_b.keys())

    if not common_terms:
        return 0.0

    # Dot product
    dot_product = sum(vec_a[term] * vec_b[term] for term in common_terms)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ─────────────────────────────────────────────
# Contextual Ad Relevance Scorer
# ─────────────────────────────────────────────

class ContextualAdScorer:
    """
    Core scoring engine that matches ads to article content
    using TF-IDF and cosine similarity.

    This is a simplified version of what contextual ad platforms
    like GumGum do at scale — analyze page content and find the
    most semantically relevant ads without using any user data.
    """

    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        self.ad_vectors = []
        self.ads = []
        self.is_fitted = False

    def fit(self, ads: list[dict]):
        """
        Fit the scorer on the ad inventory.
        Learns vocabulary and computes TF-IDF vectors for all ads.
        """
        self.ads = ads
        tokenized_ads = [preprocess(ad['content']) for ad in ads]
        self.ad_vectors = self.vectorizer.fit_transform(tokenized_ads)
        self.is_fitted = True
        print(f"✅ Fitted scorer on {len(ads)} ads | Vocabulary size: {len(self.vectorizer.vocabulary):,} terms")
        return self

    def score(self, article: dict, top_k: int = 5) -> pd.DataFrame:
        """
        Score all ads against an article and return top_k results.

        Args:
            article: dict with 'title' and 'content' keys
            top_k: number of top ads to return

        Returns:
            DataFrame with ranked ads and relevance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score()")

        # Preprocess and vectorize the article
        article_text = article['title'] + ' ' + article['content']
        article_tokens = preprocess(article_text)
        article_vector = self.vectorizer.transform(article_tokens)

        # Score each ad
        results = []
        for ad, ad_vector in zip(self.ads, self.ad_vectors):
            score = cosine_similarity(article_vector, ad_vector)

            # Find top matching keywords for explainability
            common_terms = set(article_vector.keys()) & set(ad_vector.keys())
            top_keywords = sorted(
                common_terms,
                key=lambda t: article_vector.get(t, 0) * ad_vector.get(t, 0),
                reverse=True
            )[:5]

            results.append({
                'rank': None,
                'ad_id': ad['id'],
                'brand': ad['brand'],
                'category': ad['category'],
                'relevance_score': round(score, 4),
                'top_matching_keywords': ', '.join(top_keywords) if top_keywords else 'none',
                'ad_preview': ad['content'][:80] + '...'
            })

        # Sort by relevance score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('relevance_score', ascending=False).head(top_k)
        results_df['rank'] = range(1, len(results_df) + 1)
        results_df = results_df.reset_index(drop=True)

        return results_df

    def score_all_articles(self, articles: list[dict], top_k: int = 3) -> dict:
        """Score all articles and return a summary dict."""
        all_results = {}
        for article in articles:
            results = self.score(article, top_k=top_k)
            all_results[article['id']] = {
                'article_title': article['title'],
                'top_ads': results
            }
        return all_results


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────

def evaluate_relevance(results: dict) -> pd.DataFrame:
    """
    Compute summary evaluation metrics across all articles.

    In a real system you'd evaluate against human-labeled
    relevance judgments. Here we use the top score as a
    proxy for model confidence.
    """
    rows = []
    for article_id, data in results.items():
        top_ads = data['top_ads']
        rows.append({
            'article_id': article_id,
            'article_title': data['article_title'][:50] + '...',
            'top_ad_brand': top_ads.iloc[0]['brand'],
            'top_score': top_ads.iloc[0]['relevance_score'],
            'avg_top3_score': round(top_ads['relevance_score'].mean(), 4),
            'score_gap': round(
                top_ads.iloc[0]['relevance_score'] - top_ads.iloc[-1]['relevance_score'], 4
            )
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def print_results(article: dict, results: pd.DataFrame):
    """Pretty print results to console."""
    print("\n" + "=" * 70)
    print(f"📰 Article: {article['title']}")
    print("=" * 70)
    print(f"{'Rank':<6} {'Brand':<20} {'Category':<22} {'Score':<10} {'Keywords'}")
    print("-" * 70)
    for _, row in results.iterrows():
        print(
            f"{int(row['rank']):<6} "
            f"{row['brand']:<20} "
            f"{row['category']:<22} "
            f"{row['relevance_score']:<10} "
            f"{row['top_matching_keywords']}"
        )


def generate_html_report(all_results: dict, eval_df: pd.DataFrame) -> str:
    """
    Generate a clean HTML report of results.
    Can be saved and opened in a browser.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Contextual Ad Relevance Report</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 900px; 
               margin: 40px auto; padding: 0 20px; color: #333; }
        h1 { color: #1a1a2e; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #16213e; margin-top: 40px; }
        h3 { color: #0f3460; }
        .article-block { background: #f8f9fa; border-left: 4px solid #4CAF50; 
                         padding: 20px; margin: 20px 0; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th { background: #1a1a2e; color: white; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f5f5f5; }
        .score { font-weight: bold; color: #4CAF50; }
        .rank-1 { background: #fff9e6; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; 
                 font-size: 12px; background: #e8f5e9; color: #2e7d32; }
        .summary-box { background: #e8f5e9; padding: 15px; border-radius: 8px; 
                       margin: 20px 0; }
        .keyword { font-size: 12px; color: #666; font-style: italic; }
        footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid #eee; 
                 color: #999; font-size: 13px; }
    </style>
</head>
<body>
    <h1>🎯 Contextual Ad Relevance Scorer</h1>
    <p><strong>Author:</strong> Jessica Guo &nbsp;|&nbsp; 
       <strong>Method:</strong> TF-IDF + Cosine Similarity &nbsp;|&nbsp;
       <strong>Privacy:</strong> Zero user data — content-only matching</p>
    
    <div class="summary-box">
        <strong>What this demonstrates:</strong> Matching ads to webpage content 
        using semantic similarity — the core challenge in contextual advertising. 
        No cookies. No user tracking. Just content intelligence.
    </div>

    <h2>📊 Summary Evaluation</h2>
    <table>
        <tr>
            <th>Article</th>
            <th>Top Matched Ad</th>
            <th>Top Score</th>
            <th>Avg Top-3 Score</th>
            <th>Discrimination Gap</th>
        </tr>
    """

    for _, row in eval_df.iterrows():
        html += f"""
        <tr>
            <td>{row['article_title']}</td>
            <td><span class="badge">{row['top_ad_brand']}</span></td>
            <td class="score">{row['top_score']}</td>
            <td>{row['avg_top3_score']}</td>
            <td>{row['score_gap']}</td>
        </tr>"""

    html += """
    </table>
    <h2>🔍 Detailed Results by Article</h2>
    """

    for article_id, data in all_results.items():
        html += f"""
    <div class="article-block">
        <h3>📰 {data['article_title']}</h3>
        <table>
            <tr>
                <th>Rank</th><th>Brand</th><th>Category</th>
                <th>Relevance Score</th><th>Matching Keywords</th>
            </tr>
        """
        for _, row in data['top_ads'].iterrows():
            row_class = 'rank-1' if row['rank'] == 1 else ''
            html += f"""
            <tr class="{row_class}">
                <td>{'🥇' if row['rank'] == 1 else int(row['rank'])}</td>
                <td><strong>{row['brand']}</strong></td>
                <td>{row['category']}</td>
                <td class="score">{row['relevance_score']}</td>
                <td class="keyword">{row['top_matching_keywords']}</td>
            </tr>"""

        html += """
        </table>
    </div>"""

    html += """
    <h2>⚙️ How It Works</h2>
    <ol>
        <li><strong>Preprocessing:</strong> Article and ad text are tokenized, 
            lowercased, and stop words removed</li>
        <li><strong>TF-IDF Vectorization:</strong> Each document is converted to 
            a weighted term vector. Rare, domain-specific terms get higher weight</li>
        <li><strong>Cosine Similarity:</strong> The angle between article and ad 
            vectors measures semantic overlap — length-invariant and interpretable</li>
        <li><strong>Ranking:</strong> Ads are sorted by relevance score and top-K 
            returned with explainable keyword matches</li>
    </ol>
    
    <h2>🚀 Production Extensions</h2>
    <p>In a real contextual ad platform like GumGum, this pipeline would extend to:</p>
    <ul>
        <li>Transformer-based embeddings (BERT, sentence-transformers) for deeper 
            semantic understanding</li>
        <li>Real-time inference with sub-100ms latency for RTB auction integration</li>
        <li>Distributed processing with Spark for billions of daily page impressions</li>
        <li>A/B testing framework to compare contextual models against behavioral baselines</li>
        <li>MLOps pipeline with drift monitoring and automated retraining triggers</li>
    </ul>

    <footer>
        Built by Jessica Guo | 
        <a href="https://github.com/Jessicalllll">GitHub</a> | 
        <a href="https://jessicalllll.github.io/jessica_guo.github.io/">Portfolio</a>
    </footer>
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n🎯 Contextual Ad Relevance Scorer")
    print("=" * 70)
    print("Matching ads to articles using TF-IDF + Cosine Similarity")
    print("Zero user data — content-only matching\n")

    # Initialize and fit scorer
    scorer = ContextualAdScorer()
    scorer.fit(AD_INVENTORY)

    # Score all articles
    print(f"\n📰 Scoring {len(ARTICLES)} articles against {len(AD_INVENTORY)} ads...\n")
    all_results = scorer.score_all_articles(ARTICLES, top_k=3)

    # Print results
    for article in ARTICLES:
        results = all_results[article['id']]['top_ads']
        print_results(article, results)

    # Evaluation summary
    print("\n\n📊 EVALUATION SUMMARY")
    print("=" * 70)
    eval_df = evaluate_relevance(all_results)
    print(eval_df.to_string(index=False))

    # Save results to JSON
    json_output = {}
    for article_id, data in all_results.items():
        json_output[article_id] = {
            'article_title': data['article_title'],
            'top_ads': data['top_ads'].to_dict(orient='records')
        }

    with open('results.json', 'w') as f:
        json.dump(json_output, f, indent=2)
    print("\n✅ Results saved to results.json")

    # Generate HTML report
    html_report = generate_html_report(all_results, eval_df)
    with open('report.html', 'w') as f:
        f.write(html_report)
    print("✅ HTML report saved to report.html")
    print("\n🚀 Open report.html in your browser to see the full visualization!")


if __name__ == "__main__":
    main()
