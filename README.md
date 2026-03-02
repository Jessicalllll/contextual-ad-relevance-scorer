# Contextual Ad Relevance Scorer
A small project exploring how contextual signals from webpage content can be used to rank ads by relevance without relying on user-level tracking.
The goal is to simulate a simple contextual advertising system similar to what privacy-first ad platforms use.
## Idea
Instead of using user behavioral data, this project focuses on matching the **content of a page** with the **content of an ad**.
Example:

Page content:  
"Best running shoes for marathon training"

Relevant ad:  
"Lightweight running shoes with extra cushioning"
The system scores how well an ad matches the page context.
## Approach
The current prototype uses a simple NLP pipeline:
1. Text preprocessing
2. TF-IDF vectorization
3. Cosine similarity scoring
4. Ranking ads by contextual relevance
This is obviously a simplified model, but it demonstrates the core idea behind contextual ad targeting.
## Project Structure
contextual-ad-relevance-scorer
│

├── ad_relevance_scorer.py # main scoring logic

├── experiment_tracker.py # simple experiment logging

├── monitor.py # basic monitoring simulation

├── simulate_production.py # simulate a production scoring pipeline

└── README.md
## Example Workflow
1. Input page content
2. Extract keywords / vectorize text
3. Compare with candidate ads
4. Compute similarity score
5. Rank ads by score
   Page: “camping gear for winter trips”
   
   Ads scored:
   
   Ad A: “winter sleeping bags” → 0.82
   
   Ad B: “beach umbrellas” → 0.12
   
   Ad C: “camping tents for cold weather” → 0.77

## What This Project Demonstrates

This project was mainly built to explore a few concepts relevant to ad-tech systems:

- Contextual ad targeting

- Lightweight relevance scoring

- Basic experiment logging

- Monitoring signals for model behavior

## Possible Improvements

Some directions that could make the system more realistic:

- Using transformer embeddings instead of TF-IDF

- Adding click-through-rate prediction

- Training a learning-to-rank model

- Incorporating real ad inventory data

## Why I Built This

While preparing for interviews in ad tech, I wanted to better understand how contextual ad systems work and how relevance scoring might be implemented in practice.

This project is a simplified prototype to explore those ideas.
 
