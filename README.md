# wc-match-predictor
# ⚽ FIFA World Cup Match Predictor

An end-to-end machine learning pipeline that predicts FIFA World Cup match outcomes (home win / draw / away win) using 150 years of international football data. Trained on 49,000+ historical matches and validated against real 2026 World Cup results.

---

## 📁 Project Structure

| File | Description |
|---|---|
| `build_dataset.py` | Merges raw CSVs into a clean match dataset, separates played vs upcoming fixtures |
| `feature_engineering.py` | Builds Elo ratings, rolling form windows, and head-to-head features |
| `finalize_model.py` | Trains the final class-weighted gradient boosting model, generates predictions |
| `model_summary_stats.py` | Evaluates model with full accuracy, F1, confusion matrix, and ROC-AUC stats |
| `fixture_predictions_final.csv` | Win/draw/loss probabilities for all 52 remaining 2026 WC fixtures |
| `model_summary_stats_final.csv` | Summary metrics for the final model vs baseline |
| `train_models.py` | Reference: initial logistic regression vs gradient boosting comparison |
| `tune_model.py` | Reference: class-weighting experiment |
| `two_stage_model.py` | Reference: draw/no-draw two-stage model experiment |

---

## 🚀 How to Run

**Prerequisites**
```bash
pip install scikit-learn pandas numpy joblib
```

**Run in this order:**
```bash
python build_dataset.py
python feature_engineering.py
python finalize_model.py
python model_summary_stats.py
```

> The raw CSV files (`results.csv`, `shootouts.csv`, `goalscorers.csv`, `former_names.csv`) must be in the same directory as the scripts.

---

## 📊 Features Used

- **Elo ratings** — chronologically updated after every match, with a goal-difference multiplier
- **Rolling form** — points per game, goals scored/conceded over last 5 and 10 matches
- **Head-to-head record** — win rate between the two specific teams from prior meetings only
- **Draw-detection signals** — Elo closeness, form differential, low-scoring match patterns
- **Venue** — neutral ground flag (all World Cup games are played at neutral sites)

All features are computed using only information available *before* each match to prevent data leakage.

---

## 🧠 Model

| Model | CV Accuracy | Val Accuracy | Val Macro-F1 |
|---|---|---|---|
| Baseline (higher Elo wins) | — | 50% | 0.389 |
| Logistic Regression | — | 50% | 0.389 |
| **Final: Class-weighted HGB** | **53%** | **40%*** | **0.399** |

**Final model:** `HistGradientBoostingClassifier` (sklearn) with balanced class weights to improve draw detection. 3-class output: Home win (H), Draw (D), Away win (A).

> \* Validation set is 20 already-played 2026 World Cup group-stage matches — a small, high-stakes sample of closely matched teams. 5-fold cross-validated accuracy on the full 49,405-match training set (53%) is the more reliable performance estimate.

---

## 📂 Data

**Source:** [martj42/international_results](https://github.com/martj42/international_results) via [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

**License:** CC0 1.0 — Public Domain. No restrictions on use or redistribution.

**Coverage:**
- 49,000+ international matches from 1872 to June 2026
- Includes FIFA World Cup, continental championships, qualifiers, and friendlies
- 2026 World Cup: 20 completed group-stage matches (used as validation) + 52 upcoming fixtures (prediction targets)

---

## ⚠️ Limitations

- The model rarely predicts draws — draws are a structurally hard outcome to predict from pre-match statistics alone, even with class weighting
- "Current form" is measured across all competition types (qualifiers, friendlies, etc.), not World Cup form specifically
- A 20-game validation set is too small to draw strong conclusions — results should be interpreted with caution
- 53% accuracy on a 3-class problem is above the naive baseline but modest — roughly comparable to published football prediction benchmarks

---

## 🤖 AI Diligence Statement

This project was built with AI assistance (Claude, Anthropic) for code generation, pipeline architecture, and feature engineering decisions. All modeling choices, data sourcing, validation strategy, and final outputs were reviewed and directed by the author. The AI was used as a coding accelerator, not a replacement for understanding — the author can speak to every technical decision in this project.

---

## 🛠️ Built With

- Python 3
- scikit-learn
- pandas
- numpy
- joblib
