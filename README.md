# Actuarial Risk Pipeline — Prudential Life Insurance Assessment

## What this project is

I built this as my final year MSc Statistics project at the Pondicherry University 
for the subject Data Analysis using Statistical Packages.
The idea was to take a real insurance dataset and build 
something that actually works end to end — not just a model in a 
notebook, but a full pipeline from raw data to a live dashboard.

The dataset is from Kaggle (Prudential Life Insurance Assessment). 
It has 59,381 applicants and 128 features. The goal is to predict 
each applicant's risk score from 1 to 8.

## Why this problem matters

When an insurer gets the risk score wrong, there are real consequences. 
Score someone too low and you undercharge them — the company loses money 
when they claim. Score them too high and you lose the customer to a 
competitor. Getting this right is the core job of an underwriter, and 
this project tries to automate that decision in a defensible, 
explainable way.

## What I was aiming for

- Quadratic Weighted Kappa above 0.60 on unseen test data
- Model predictions that stay within 5% of observed claim rates
- Something I can actually open and demo in an interview

## How it's structured

- Phase 1 — Data Engineering (SQL, DB Browser for SQLite)
- Phase 2 — Statistical Analysis and Multivariate Methods (R, RStudio)
- Phase 3 — Machine Learning with XGBoost (Python)
- Phase 4 — Monte Carlo Simulation using a Gaussian Copula (Python)
- Phase 5 — Interactive Dashboard (Python, Streamlit)

## Tools used

Python 3.13, R 4.4, SQLite, RStudio, Streamlit, XGBoost, SHAP, 
scikit-learn, tidyverse, FactoMineR

## Author

Priyanka Choudhury
MSc Statistics — University of Pondicherry, 2026
