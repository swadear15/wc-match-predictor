# ⚽ FIFA World Cup Match & Playoff Predictor

An end-to-end machine learning pipeline that (1) predicts international match outcomes
(home win / draw / away win) from 150 years of results and (2) uses a World Cup **group stage to
predict the knockout rounds**, by simulating the full bracket. Everything is backtested the honest way: train on the
past, predict the future, compare with what actually happened in the 2026 World Cup.

---

## 🆕 What changed in v2

| | v1 (original) | v2 |
|---|---|---|
| Evaluation | shuffled 5-fold CV + 20-match validation | **time-ordered backtest** on World Cups 2010-2026 (360 matches) and 2,712 competitive matches 2023-26 |
| Draw handling | class re-weighting (forces ~25% draw picks) | **no re-weighting**; calibrated W/D/L probabilities, pick = most likely outcome |
| Elo | fixed K=20 | K by match importance, explicit home advantage, opponent-adjusted form, goal rates, rest days |
| Models | one gradient-boosting classifier | ensemble: logistic + gradient boosting + Poisson goal model |
| Scorelines | none | Poisson/Dixon-Coles goal model (needed for group tie-breaks) |
| Playoffs | none | **group stage → knockout model + 2026 bracket Monte Carlo** |
| Data | snapshot with 20 of 72 group games played | refreshed to Oct 2026: the whole tournament, incl. knockouts |

### Results (all out-of-sample)

**Match outcomes, World Cups 2010-2026 pooled (n = 360):**

| Model | Accuracy | Log-loss ↓ | RPS ↓ |
|---|---|---|---|
| v1 original (class-weighted GBM) | 49.2% | 0.992 | 0.201 |
| v1 features, no class weights | 57.2% | 0.946 | 0.191 |
| Elo-only logistic | 57.8% | 0.947 | 0.192 |
| **v2 ensemble** | **58.3%** | 0.948 | 0.192 |

**Competitive matches 2023-26, excluding friendlies (n = 2,712):** v1 58.4% / 0.878 → Elo-only 61.4% / 0.853 → **v2 61.5% / 0.843** (RPS 0.171 → 0.165).

**The 52 group games that were still unplayed in the original data snapshot** (model fit only on matches before 2026-06-11): v2 **65.4%** accuracy (log-loss 0.822) vs v1 **53.8%** (0.868). All 72 group games: 61.1% vs 51.4%.

What this says, plainly:
- The big win is **removing class re-weighting**: +8 points on World Cups from that change alone (49.2% → 57.2%), +9 with the full v2 ensemble. Re-weighting made v1 call a draw in ~28% of games and distorted its probabilities.
- The richer v2 features/ensemble add a **small but consistent** gain over plain Elo on the large 2023-26 sample (log-loss 0.853 → 0.843) and are **indistinguishable from Elo on the 360 World Cup games.** Football outcomes are noisy, and ~58-62% three-way accuracy appears to be close to what models built from results alone can reach. The old README's "53% CV / 40% val" are not comparable to these numbers (shuffled CV, 20 games).

Per-edition detail: `backtest_match_models.csv`. Group-stage picks vs reality: `group_stage_predictions_2026.csv`.

### Group stage → playoffs

`knockout_model.py` fits P(team advances) on every World Cup knockout game since 1986 (150 games; 3rd-place games excluded) and asks which group-stage information is worth using. Leave-one-World-Cup-out results:

| Predictor | LOEO log-loss ↓ | 2026 hold-out log-loss ↓ |
|---|---|---|
| Pre-tournament Elo gap (ignores the group stage) | 0.625 | 0.447 |
| **Elo gap after the group stage** (Elo updated by every group game) | **0.590** | **0.419** |
| + group goal difference / points / goals / over-performance / "won group" | 0.583 – 0.592 | 0.419 – 0.446 |

(coin flip = 0.693.) **The group stage is informative, and it is already captured by updating Elo through it**; adding explicit group statistics on top does not reliably help, so the final model is the simple one: `P(advance) = sigmoid(0.78 × Elo gap / 100)`. On the same 91 knockout games from 2010-2026 it beats the match model repurposed for knockouts (log-loss 0.489 vs 0.547).

`simulate_playoffs.py` turns that into a full 2026 bracket (12 groups, 8 best third-placed teams, round of 32 → final), with 10,000 simulations per scenario. Backtest vs. the real tournament (Spain beat Argentina 1-0 in the final):

| Scenario | P(Spain champion) | Brier skill vs naive* |
|---|---|---|
| A. Pre-tournament (all 104 games simulated) | 26.1% (rank 1; Argentina 20.6%) | 0.41 |
| B. After the group stage (real group results, real bracket) | 23.4% (rank 2; Argentina 29.0%) | **0.49** |
| C. Same bracket, ignoring the group stage (pre-tournament Elo) | 23.2% (rank 1) | 0.44 |

\*mean over the stages R16, QF, SF, Final, Champion (and R32 for A) of `1 − Brier/Brier_naive`, where "naive" gives every eligible team the same probability for a stage.

On the 31 real knockout games: the knockout model scored log-loss **0.419** (accuracy 24/31) vs 0.447 (25/31) for pre-tournament Elo, 0.484 (25/31) for the v2 match model and 0.515 (27/31) for the original v1. Accuracy differences of 1-3 games out of 31 are noise (a handful of near coin-flip games decided them); the proper scoring rules (log-loss, Brier) favour the new model. Files: `knockout_predictions_2026.csv`, `knockout_match_comparison_2026.csv`, `playoff_probabilities_*.csv`, `playoff_backtest_summary.csv`.

---

## 📁 Project Structure

| File | Description |
|---|---|
| `build_dataset.py` | Merges raw CSVs into a clean match dataset (played vs upcoming, shootout winners) |
| `feature_engineering.py` | **v1** Elo / form / head-to-head features (kept as the baseline for the backtest) |
| `features_v2.py` | **v2** features: importance-weighted Elo, home advantage, opponent-adjusted form, goal rates, rest, h2h |
| `goal_model.py` | Poisson goal model with Dixon-Coles correction (scorelines for group tie-breaks) |
| `match_model_v2.py` | v2 W/D/L ensemble; `backtest`, `train` and `groups2026` commands |
| `tournament.py` | Groups, FIFA 2026 tie-breakers, round-of-32 bracket, historical group → knockout dataset |
| `knockout_model.py` | The group stage → playoffs model and its feature comparison |
| `simulate_playoffs.py` | Monte Carlo bracket simulation + backtest against the real 2026 results |
| `finalize_model.py`, `model_summary_stats.py`, `train_models.py`, `tune_model.py`, `two_stage_model.py` | v1 scripts, kept for reference |
| `fixture_predictions*.csv`, `model_summary_stats_final.csv`, `validation_predictions*.csv` | v1 outputs from the June 2026 snapshot (superseded) |

## 🚀 How to Run

```bash
pip install scikit-learn pandas numpy scipy joblib
python build_dataset.py            # -> matches_played.csv, fixtures_upcoming.csv
python feature_engineering.py      # -> matches_featured.csv (only needed for the v1 rows of the backtest)
python match_model_v2.py backtest  # -> backtest_match_models.csv
python match_model_v2.py groups2026   # -> group_stage_predictions_2026.csv
python knockout_model.py           # -> knockout_model_cv.csv, knockout_model.json
python simulate_playoffs.py --sims 10000   # -> playoff_*.csv, knockout_*_2026.csv
```
The raw CSVs (`results.csv`, `shootouts.csv`, `goalscorers.csv`, `former_names.csv`) must be in the same directory. Total runtime is roughly 10 minutes (the backtest is the slow part).

## 📊 Features Used (v2)

- **Elo** — chronologically updated, K = 60 (World Cup) / 50 (continental finals) / 40 (qualifiers, Nations League) / 30 (other tournaments) / 20 (friendlies), goal-difference multiplier, +100 for a non-neutral home side
- **Form** — points, goals for/against over the last 5 and 10 games; opponent-adjusted "surprise" (actual − Elo-expected); Elo trend; exponentially-weighted attack/defence rates
- **Context** — rest days, matches played, match importance, shrunk head-to-head record, home flag

Every feature is computed from matches strictly **before** the one being predicted.

## 📂 Data

**Source:** [martj42/international_results](https://github.com/martj42/international_results) via [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — CC0 1.0, public domain.
**Coverage:** ~49,900 international matches, 1872 to October 2026, including the complete 2026 World Cup (72 group games + 32 knockout games). The data files were refreshed from upstream for v2; the v1 outputs reflect the earlier snapshot.

## ⚠️ Limitations

- **Draws remain a coin the model can't call.** It reports P(draw) (about 22% on average, never above ~30% for the 2026 group games; the real draw rate in that group stage was 28%), but a draw is almost never the single most likely outcome, so picks are essentially win/lose.
- **Football is noisy.** Differences between v2 and plain Elo on the 360 World Cup games are well inside the noise; don't read the 58.3% vs 57.8% gap as a real improvement.
- **The playoff backtest is one tournament.** The champion probabilities (e.g. Spain 26%) can't be calibrated from a single event, and Elo-based models tend to concentrate probability on the top teams. The 31-game knockout comparison is small.
- **Third-place bracket slots are approximate.** FIFA assigns third-placed teams to round-of-32 slots from a fixed 495-row table that isn't in the dataset. The simulator draws uniformly among the valid assignments (22 for the real 2026 combination). Group winner/runner-up pairings are exact; when testing from the actual group results, the real pairings are used.
- **Simplifications:** knockout venues are treated as neutral (no host advantage after the group stage); fair-play and FIFA-ranking tie-breakers are replaced by a random draw; extra time and penalties are folded into a single P(advance); squad quality, injuries and market values are not available in this dataset.
- Historical team renames (`former_names.csv`) are not merged into the Elo history.

---

## 🤖 AI Diligence Statement

This project was built with AI assistance (Claude, Anthropic) for code generation, pipeline architecture, and feature engineering decisions. All modeling choices, data sourcing, validation strategy, and final outputs were reviewed and directed by the author. The AI was used as a coding accelerator, not a replacement for understanding — the author can speak to every technical decision in this project.

The v2 update (time-ordered backtest, ensemble, group stage → playoff model and simulator) was likewise produced with Claude (Claude Code); its results should be reviewed and re-run by the author before being cited.

## 🛠️ Built With

Python 3 · scikit-learn · pandas · numpy · scipy · joblib
