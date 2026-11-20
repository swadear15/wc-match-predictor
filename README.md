# ⚽ FIFA World Cup Match & Playoff Predictor

An end-to-end machine learning pipeline that (1) predicts international match outcomes
(home win / draw / away win) from 150 years of results and (2) uses a World Cup **group stage and
previous World Cups to predict the knockout bracket**, treating each knockout game as a win/lose
classification (no draws in the playoffs). Everything is backtested the honest way: train on the
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

## 🏆 Knockout bracket classifier (v3)

Knockout games can't end in a draw (extra time and penalties decide them), so `playoff_classifier.py`
treats each one as a **binary classification: does team 1 advance?** It is trained on every World Cup
knockout game from 1986-2022 (150 games, third-place games excluded) and tested on the 31 knockout
games of 2026, which it never sees during training.

**Features are frozen at the end of the group stage**, which is all you know when you fill in a bracket.
A quarter-final row does not see the round-of-16 results. All 15 candidates are "team 1 minus team 2":
- **Strength:** Elo before the tournament, Elo after the group stage, Elo gained in the group stage
- **Group stage:** points, goal difference, goals for and against per game, over-performance vs Elo, group winner, finishing position
- **Previous World Cups:** games played at the last edition (how far they went), games and win rate over the last 3 editions, titles won
- **Venue:** host nation

Every model is trained on both orientations of each game, with symmetrised predictions, so
P(A beats B) = 1 − P(B beats A) exactly.

| Model | Features | LOEO log-loss ↓ | 2026 log-loss ↓ | 2026 games correct | 2026 bracket (pts / 80) |
|---|---|---|---|---|---|
| **Logistic: Elo after group stage (selected)** | 1 | 0.594 | **0.410** | 25 / 31 | 55 |
| Logistic: pre-tournament Elo (ignores group stage) | 1 | 0.625 | 0.448 | 25 / 31 | 70 |
| Logistic: group stage only (no Elo) | 4 | **0.587** | 0.531 | 24 / 31 | 38 |
| Logistic: Elo + group stage | 5 | 0.589 | 0.475 | 23 / 31 | 43 |
| Logistic: Elo + previous World Cups | 4 | 0.600 | 0.461 | 24 / 31 | 43 |
| Logistic: all 15 features | 15 | 0.603 | 0.518 | 21 / 31 | 40 |
| Gradient boosting: all features | 15 | 0.616 | 0.497 | 25 / 31 | 46 |
| Random forest: all features | 15 | 0.593 | 0.499 | 24 / 31 | 43 |

LOEO = leave-one-World-Cup-out on 1986-2022. Coin flip = log-loss 0.693 and an expected 15.5 / 80 bracket
points. Bracket scoring: each round is worth 16 points (1-2-4-8-16 per correct pick), and a pick only scores
if that team actually won the game in that slot. Full table: `playoff_classifier_results.csv`.

**What the data says**
- **The group stage carries real signal.** A model that sees only group results (no ratings at all) is as good as
  the Elo models, and Elo updated through the group stage beats pre-tournament Elo on both the 1986-2022
  cross-validation and on 2026.
- **150 games can't separate the top models.** Eight of the ten are within one standard error of the best
  cross-validated log-loss (paired, per game). So the classifier uses the one-standard-error rule: the simplest
  model that is statistically tied with the best, which is one feature, Elo after the group stage. *Caveat: I set
  this rule after seeing the 2026 results, so the 2026 numbers are not a fully blind test.* The repo's earlier
  `knockout_model.py` used a similar "simplest within 0.01" rule, which lands on the same model.
- **Extra features and non-linear models don't help.** Previous-World-Cup history, host status, all 15 features,
  gradient boosting and random forests are all no better than one Elo number, and they are worse on 2026. With
  150 training games the extra flexibility mostly fits noise.

**The selected model's 2026 bracket** (`bracket_prediction_2026.csv`), filled in from the group stage alone:
13/16 round-of-32 picks, 5/8 round of 16, **4/4 quarter-finals, 2/2 semi-finals** (it named the final four,
France, Spain, England and Argentina, and the final, Spain vs Argentina), then picked Argentina (53%) in a final
that Spain won. **55 of 80 points.** Pre-tournament Elo scored more (70) only because it leaned Spain in the final.
The bracket score of a single tournament mostly comes down to one or two close games, which is why the model is
chosen on log-loss, not bracket points.

Exact probabilities of reaching each round (`round_probabilities_2026.csv`, computed over the bracket tree rather than simulated):
Argentina 33% to win, Spain 25%, France 20%, England 6%.
Every game's prediction: `knockout_game_predictions_2026.csv`.

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
| `playoff_classifier.py` | **v3** knockout classifier: features frozen after the group stage + previous World Cups, model comparison, 2026 bracket prediction and scoring |
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
python playoff_classifier.py      # -> playoff_classifier_results.csv, bracket_prediction_2026.csv, round_probabilities_2026.csv
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
