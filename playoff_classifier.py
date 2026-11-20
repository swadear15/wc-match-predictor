"""
playoff_classifier.py -- predict the World Cup knockout bracket from the group stage.

Knockout games cannot end in a draw (extra time / penalties decide them), so each game is a
binary classification: does team 1 advance?  y = 1 / 0.

    python playoff_classifier.py

Data   : every World Cup knockout game 1986-2022 (training, 150 games; 3rd-place games
         excluded) and the 31 knockout games of 2026 (held out, never used for fitting).
Features are FROZEN AT THE END OF THE GROUP STAGE, i.e. exactly what is known when the
bracket is filled in. A quarter-final row does NOT see what happened in the round of 16.
Every feature is a difference "team 1 minus team 2":

  strength   d_elo_post   Elo after the last group game (/100)
             d_elo_pre    Elo before the tournament (/100)
             d_elo_gain   Elo gained during the group stage (/100)
  group      d_ppg, d_gd_pg, d_gf_pg, d_ga_pg   points / goal diff / goals for / against per game
             d_overperf   points share above what Elo expected, per game
             d_won_group, d_rank                won the group; finishing position (1-4)
  history    d_prev_games     World Cup games played at the previous edition (0-7: how far they went)
             d_prev3_games    ... summed over the previous 3 editions
             d_prev3_winrate  win rate in World Cup games over the previous 3 editions
             d_titles         log(1 + World Cups won before this edition)
  venue      d_host           tournament host

Models are trained on both orientations of every game (A vs B and B vs A) and their
predictions are symmetrised, so P(A beats B) = 1 - P(B beats A) exactly.

Evaluation
  * leave-one-World-Cup-out (LOEO) on 1986-2022 -> used to pick the model
  * the picked model is then fit on 1986-2022 and scored on 2026, game by game and as a
    whole bracket (a predicted team must have survived your earlier picks to score).
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from features_v2 import build_features, gd_multiplier, k_factor, load_played
from tournament import advancing_team, edition, group_stage_stats, shootout_map

warnings.filterwarnings('ignore')

HOSTS = {1986: {'Mexico'}, 1990: {'Italy'}, 1994: {'United States'}, 1998: {'France'},
         2002: {'South Korea', 'Japan'}, 2006: {'Germany'}, 2010: {'South Africa'},
         2014: {'Brazil'}, 2018: {'Russia'}, 2022: {'Qatar'},
         2026: {'United States', 'Mexico', 'Canada'}}
TRAIN_YEARS = list(range(1986, 2023, 4))
TEST_YEAR = 2026
ROUND_NAMES = {32: ['Round of 32', 'Round of 16', 'Quarter-final', 'Semi-final', 'Final'],
               16: ['Round of 16', 'Quarter-final', 'Semi-final', 'Final']}

FEATURES = ['d_elo_post', 'd_elo_pre', 'd_elo_gain', 'd_ppg', 'd_gd_pg', 'd_gf_pg', 'd_ga_pg',
            'd_overperf', 'd_won_group', 'd_rank', 'd_prev_games', 'd_prev3_games',
            'd_prev3_winrate', 'd_titles', 'd_host']


# ----------------------------------------------------------------------------- history
def wc_history(d):
    """Per edition: games played, games won, champion (winner of the final)."""
    w = d[d.tournament == 'FIFA World Cup']
    smap = shootout_map()
    hist = {}
    for yr, g in w.groupby(w.date.dt.year):
        g = g.sort_values('date', kind='stable')
        games, wins = {}, {}
        for r in g.itertuples():
            for t in (r.home_team, r.away_team):
                games[t] = games.get(t, 0) + 1
            win = r.home_team if r.home_score > r.away_score else (
                r.away_team if r.away_score > r.home_score else None)
            if win:
                wins[win] = wins.get(win, 0) + 1
        champ = None if yr == 1950 else advancing_team(g.iloc[-1], smap)   # 1950 had no final
        hist[yr] = dict(games=games, wins=wins, champ=champ)
    return hist


def history_features(hist, team, year):
    prev = [y for y in sorted(hist) if y < year]
    last3 = prev[-3:]
    g3 = sum(hist[y]['games'].get(team, 0) for y in last3)
    w3 = sum(hist[y]['wins'].get(team, 0) for y in last3)
    return dict(prev_games=hist[prev[-1]]['games'].get(team, 0) if prev else 0,
                prev3_games=g3, prev3_winrate=w3 / g3 if g3 else 0.0,
                titles=np.log1p(sum(hist[y]['champ'] == team for y in prev)))


# ---------------------------------------------------------------------- team snapshots
def team_snapshots(d, year, hist):
    """State of every team in an edition at the END of the group stage."""
    gs, _ = edition(d, year)
    st = group_stage_stats(gs)
    elo_pre, elo_post = {}, {}
    k = k_factor('FIFA World Cup')
    for r in gs.itertuples():
        elo_pre.setdefault(r.home_team, r.elo_h)
        elo_pre.setdefault(r.away_team, r.elo_a)
        w = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
        delta = k * gd_multiplier(abs(r.home_score - r.away_score)) * (w - r.we_h)
        elo_post[r.home_team] = r.elo_h + delta     # overwritten each game -> after the last one
        elo_post[r.away_team] = r.elo_a - delta
    snap = {}
    for t, s in st.items():
        gp = max(s['gp'], 1)
        snap[t] = dict(elo_post=elo_post[t] / 100, elo_pre=elo_pre[t] / 100,
                       elo_gain=(elo_post[t] - elo_pre[t]) / 100,
                       ppg=s['pts'] / gp, gd_pg=s['gd'] / gp, gf_pg=s['gf'] / gp,
                       ga_pg=(s['gf'] - s['gd']) / gp, overperf=s['overperf'] / gp,
                       won_group=float(s['rank'] == 1), rank=float(s['rank']),
                       host=float(t in HOSTS.get(year, set())),
                       **history_features(hist, t, year))
    return snap


def pair_features(snap, a, b):
    return {f'd_{k}': snap[a][k] - snap[b][k] for k in snap[a]}


# ----------------------------------------------------------------------------- bracket
def bracket_structure(d, year, smap):
    """Reconstruct an edition's knockout tree from its results (3rd-place game excluded).
    Returns rounds: list of lists of matches; a first-round match is (teamA, teamB), a later
    match is (feeder_index_A, feeder_index_B) into the previous round. Also the actual winners."""
    _, ko = edition(d, year)
    ko = ko.reset_index(drop=True)
    ko = ko.drop(index=len(ko) - 2).reset_index(drop=True)      # third-place game
    n_first = (len(ko) + 1) // 2
    rounds, winners, i, size = [], [], 0, n_first
    prev_w = None
    while size >= 1:
        rows = ko.iloc[i:i + size]
        rnd, wins = [], []
        for r in rows.itertuples():
            w = advancing_team(r, smap)
            if prev_w is None:
                rnd.append((r.home_team, r.away_team))
            else:
                fa = prev_w.index(r.home_team)
                fb = prev_w.index(r.away_team)
                rnd.append((fa, fb))
            wins.append(w)
        rounds.append(rnd)
        winners.append(wins)
        prev_w, i, size = wins, i + size, size // 2
    return rounds, winners


def predict_bracket(rounds, prob):
    """Pick the more likely team in every game, carrying your own picks forward."""
    picks, prev = [], None
    for k, rnd in enumerate(rounds):
        cur = []
        for m in rnd:
            a, b = (m if k == 0 else (prev[m[0]], prev[m[1]]))
            cur.append(a if prob(a, b) >= 0.5 else b)
        picks.append(cur)
        prev = cur
    return picks


def round_probabilities(rounds, prob):
    """Exact P(team wins each round) by dynamic programming over the bracket tree."""
    dist_prev, out = None, []
    for k, rnd in enumerate(rounds):
        dists = []
        for m in rnd:
            if k == 0:
                A, B = {m[0]: 1.0}, {m[1]: 1.0}
            else:
                A, B = dist_prev[m[0]], dist_prev[m[1]]
            dist = {}
            for a, pa in A.items():
                dist[a] = pa * sum(pb * prob(a, b) for b, pb in B.items())
            for b, pb in B.items():
                dist[b] = pb * sum(pa * prob(b, a) for a, pa in A.items())
            dists.append(dist)
        out.append({t: p for dd in dists for t, p in dd.items()})
        dist_prev = dists
    return out


def score_bracket(picks, winners):
    """Correct picks per round. Each round is worth 16 points (2026: 1-2-4-8-16 per correct pick)."""
    pts = [16 // len(w) for w in winners]          # every round is worth 16 points in total
    correct = [sum(p == w for p, w in zip(pk, wn)) for pk, wn in zip(picks, winners)]
    return correct, sum(c * p for c, p in zip(correct, pts)), sum(len(w) * p for w, p in zip(winners, pts))


def expected_coinflip_score(winners):
    pts = [16 // len(w) for w in winners]
    return sum(len(w) * p * 0.5 ** (k + 1) for k, (w, p) in enumerate(zip(winners, pts)))


# -------------------------------------------------------------------------------- data
def build_dataset():
    played = load_played('.')
    d, _ = build_features(played)
    smap = shootout_map()
    hist = wc_history(d)
    rows, snaps, brackets = [], {}, {}
    for yr in TRAIN_YEARS + [TEST_YEAR]:
        snap = team_snapshots(d, yr, hist)
        snaps[yr] = snap
        rounds, winners = bracket_structure(d, yr, smap)
        brackets[yr] = (rounds, winners)
        names = ROUND_NAMES[len(rounds[0]) * 2]
        teams_prev = None
        for k, (rnd, wins) in enumerate(zip(rounds, winners)):
            for m, w in zip(rnd, wins):
                a, b = m if k == 0 else (teams_prev[m[0]], teams_prev[m[1]])
                rows.append(dict(year=yr, round=names[k], team1=a, team2=b, winner=w,
                                 y=int(w == a), **pair_features(snap, a, b)))
            teams_prev = wins
    return pd.DataFrame(rows), snaps, brackets


# ------------------------------------------------------------------------------ models
class SymmetricModel:
    """Wrap a binary classifier: train on mirrored data, symmetrise predictions."""

    def __init__(self, kind, cols, C=1.0):
        self.kind, self.cols, self.C = kind, cols, C

    def _new(self):
        if self.kind == 'logit':
            return LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000)
        if self.kind == 'gbm':
            return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=2,
                                                  min_samples_leaf=20, l2_regularization=1.0,
                                                  random_state=0)
        if self.kind == 'rf':
            return RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.5,
                                          random_state=0, n_jobs=-1)
        raise ValueError(self.kind)

    def fit(self, df):
        X = df[self.cols].values.astype(float)
        self.scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)   # no centring: keeps symmetry
        X = X / self.scale
        y = df.y.values
        self.m = self._new().fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]))
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float)) / self.scale
        p, q = self.m.predict_proba(X)[:, 1], self.m.predict_proba(-X)[:, 1]
        return (p + 1 - q) / 2

    def predict_df(self, df):
        return self.predict(df[self.cols].values)


class Ensemble:
    def __init__(self, members):
        self.members = members

    def fit(self, df):
        for m in self.members:
            m.fit(df)
        return self

    def predict_df(self, df):
        return np.mean([m.predict_df(df) for m in self.members], axis=0)


ELO = ['d_elo_post']
GROUP = ['d_ppg', 'd_gd_pg', 'd_overperf', 'd_won_group']
HISTORY = ['d_prev_games', 'd_prev3_winrate', 'd_titles']


def candidates():
    return {
        'logit: Elo after groups': lambda: SymmetricModel('logit', ELO),
        'logit: pre-tournament Elo': lambda: SymmetricModel('logit', ['d_elo_pre']),
        'logit: group stage only (no Elo)': lambda: SymmetricModel('logit', GROUP, C=0.3),
        'logit: Elo + group stage': lambda: SymmetricModel('logit', ELO + GROUP, C=0.3),
        'logit: Elo + previous World Cups': lambda: SymmetricModel('logit', ELO + HISTORY, C=0.3),
        'logit: Elo + group + previous WCs + host': lambda: SymmetricModel('logit', ELO + GROUP + HISTORY + ['d_host'], C=0.3),
        'logit: all 15 features': lambda: SymmetricModel('logit', FEATURES, C=0.1),
        'gradient boosting: all features': lambda: SymmetricModel('gbm', FEATURES),
        'random forest: all features': lambda: SymmetricModel('rf', FEATURES),
        'ensemble: logit(Elo+group+history) + RF': lambda: Ensemble([
            SymmetricModel('logit', ELO + GROUP + HISTORY + ['d_host'], C=0.3),
            SymmetricModel('rf', FEATURES)]),
    }


def metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(n=len(y), accuracy=float(((p > .5) == y).mean()), log_loss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(np.mean((p - y) ** 2)))


def make_prob(model, snap):
    cache = {}

    def prob(a, b):
        if (a, b) not in cache:
            row = pd.DataFrame([pair_features(snap, a, b)])
            cache[(a, b)] = float(model.predict_df(row)[0])
        return cache[(a, b)]
    return prob


def main():
    D, snaps, brackets = build_dataset()
    tr_all, te = D[D.year.isin(TRAIN_YEARS)], D[D.year == TEST_YEAR]
    print(f'{len(tr_all)} training games (World Cups 1986-2022), {len(te)} held-out games (2026)')

    rows, losses = [], {}
    for name, make in candidates().items():
        # ---- leave-one-World-Cup-out on 1986-2022 (model selection uses ONLY this)
        p = pd.Series(index=tr_all.index, dtype=float)
        pts = maxpts = coin = 0
        for yr in TRAIN_YEARS:
            fit = make().fit(tr_all[tr_all.year != yr])
            sub = tr_all[tr_all.year == yr]
            p.loc[sub.index] = fit.predict_df(sub)
            rounds, winners = brackets[yr]
            _, s, mx = score_bracket(predict_bracket(rounds, make_prob(fit, snaps[yr])), winners)
            pts, maxpts, coin = pts + s, maxpts + mx, coin + expected_coinflip_score(winners)
        cv = metrics(tr_all.y.values, p.values)
        pc, yv = np.clip(p.values, 1e-4, 1 - 1e-4), tr_all.y.values
        losses[name] = -(yv * np.log(pc) + (1 - yv) * np.log(1 - pc))
        # ---- fit on 1986-2022, predict 2026
        fit = make().fit(tr_all)
        pt = fit.predict_df(te)
        ho = metrics(te.y.values, pt)
        rounds, winners = brackets[TEST_YEAR]
        picks = predict_bracket(rounds, make_prob(fit, snaps[TEST_YEAR]))
        correct, s26, mx26 = score_bracket(picks, winners)
        rows.append({'model': name,
                     'LOEO_log_loss': cv['log_loss'], 'LOEO_accuracy': cv['accuracy'], 'LOEO_brier': cv['brier'],
                     'LOEO_bracket_points_pct': pts / maxpts,
                     '2026_log_loss': ho['log_loss'], '2026_accuracy': ho['accuracy'], '2026_brier': ho['brier'],
                     '2026_bracket_points': s26, '2026_bracket_max': mx26,
                     '2026_correct_by_round': '-'.join(map(str, correct)),
                     '2026_champion_pick': picks[-1][0]})
        print(f"  {name:45s} LOEO ll={cv['log_loss']:.3f} acc={cv['accuracy']:.3f} | "
              f"2026 ll={ho['log_loss']:.3f} acc={ho['accuracy']:.3f} bracket={s26}/{mx26} ({'-'.join(map(str, correct))}) champ={picks[-1][0]}",
              flush=True)
    res = pd.DataFrame(rows)
    # one-standard-error rule: among models whose LOEO log-loss is within 1 SE (paired, per game)
    # of the best, take the one with the fewest features.
    top = min(losses, key=lambda k: losses[k].mean())
    n_feats = {n: len(getattr(candidates()[n](), 'cols', FEATURES)) for n in losses}
    se = {n: np.std(losses[n] - losses[top], ddof=1) / np.sqrt(len(losses[n])) for n in losses}
    res['LOEO_diff_vs_best'] = [losses[n].mean() - losses[top].mean() for n in res.model]
    res['LOEO_diff_1SE'] = [se[n] for n in res.model]
    res['n_features'] = [n_feats[n] for n in res.model]
    tied = [n for n in losses if losses[n].mean() - losses[top].mean() <= se[n]]
    best = min(tied, key=lambda n: (n_feats[n], losses[n].mean()))
    res['selected'] = res.model == best
    res.round(4).to_csv('playoff_classifier_results.csv', index=False)
    print(f'\nLowest LOEO log-loss: {top}. Statistically tied (within 1 SE): {len(tied)} models.')
    print(f'Selected (simplest tied model): {best}   (coin flip: log-loss {np.log(2):.3f}, '
          f'expected bracket {expected_coinflip_score(brackets[TEST_YEAR][1]):.1f}/{res["2026_bracket_max"].iloc[0]})')

    # ---- final outputs for the picked model
    model = candidates()[best]().fit(tr_all)
    snap = snaps[TEST_YEAR]
    prob = make_prob(model, snap)
    rounds, winners = brackets[TEST_YEAR]
    picks = predict_bracket(rounds, prob)
    names = ROUND_NAMES[32]

    # predicted vs actual bracket, game by game (teams as YOU would have picked them)
    out, prev_p, prev_a = [], None, None
    for k, (rnd, pk, wn) in enumerate(zip(rounds, picks, winners)):
        for i, m in enumerate(rnd):
            pa, pb = m if k == 0 else (prev_p[m[0]], prev_p[m[1]])
            aa, ab = m if k == 0 else (prev_a[m[0]], prev_a[m[1]])
            out.append(dict(round=names[k], predicted_matchup=f'{pa} vs {pb}',
                            predicted_winner=pk[i], p_predicted_winner=round(max(prob(pa, pb), prob(pb, pa)), 3),
                            actual_matchup=f'{aa} vs {ab}', actual_winner=wn[i], correct=pk[i] == wn[i]))
        prev_p, prev_a = pk, wn
    pd.DataFrame(out).to_csv('bracket_prediction_2026.csv', index=False)

    # each actual game, with the probability the model gave it
    te_out = te[['round', 'team1', 'team2', 'winner']].copy()
    te_out['p_team1_advances'] = model.predict_df(te).round(3)
    te_out['predicted_winner'] = np.where(te_out.p_team1_advances >= .5, te_out.team1, te_out.team2)
    te_out['correct'] = te_out.predicted_winner == te_out.winner
    te_out.to_csv('knockout_game_predictions_2026.csv', index=False)

    # exact probability of reaching each round
    rp = round_probabilities(rounds, prob)
    teams = sorted({t for m in rounds[0] for t in m})
    reached = {t: 0 for t in teams}
    for k, wn in enumerate(winners):
        for t in wn:
            reached[t] = k + 1
    stage = ['Round of 32', 'Round of 16', 'Quarter-final', 'Semi-final', 'Final', 'Champion']
    tab = pd.DataFrame({'team': teams})
    for k, col in enumerate(['P_round_of_16', 'P_quarter_final', 'P_semi_final', 'P_final', 'P_champion']):
        tab[col] = [rp[k].get(t, 0.0) for t in teams]
    tab['actual_result'] = [stage[reached[t]] if reached[t] < 5 else 'Champion' for t in teams]
    tab = tab.sort_values('P_champion', ascending=False)
    tab.round(4).to_csv('round_probabilities_2026.csv', index=False)

    json.dump({'model': best, 'trained_on': '1986-2022 World Cup knockout games',
               'features': getattr(model, 'cols', None)}, open('playoff_classifier_model.json', 'w'), indent=2)
    print(tab.head(10).round(3).to_string(index=False))
    print('\nBracket (picked model):')
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main()
