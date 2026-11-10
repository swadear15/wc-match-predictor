"""
simulate_playoffs.py -- Monte Carlo of the 2026 World Cup: group stage -> knockout bracket.

    python simulate_playoffs.py                 # run the full backtest + write CSVs
    python simulate_playoffs.py --sims 20000

How it works
  1. Ratings: Elo / goal rates as they stood on the `as_of` date (see features_v2.py).
  2. Group games not yet played are sampled from the Poisson goal model (goal_model.py), with
     Elo updated after every simulated game. Played games are fixed at their real scores.
  3. Group tables use the FIFA 2026 tie-breakers; the 8 best third-placed teams advance
     (tournament.py). Round-of-32 pairings follow the official bracket.
  4. Knockout games use the knockout model (knockout_model.py): P(advance) =
     sigmoid(beta * Elo gap / 100). Elo keeps updating through the knockout rounds.

Backtest (everything fit only on data available at the time):
  A. "pre-tournament": simulate all 104 games from the eve of the tournament.
  B. "after groups": real group results are known; simulate only the knockouts, rating teams
     with Elo that has been updated by the group stage.
  C. "after groups, ignoring them": same bracket but with pre-tournament Elo -> isolates the
     value of the group-stage information.
"""
import argparse
import random
import numpy as np
import pandas as pd

from features_v2 import HOME_ADV, RatingState, build_features, gd_multiplier, load_played
from goal_model import GoalModel
from knockout_model import fit_symmetric, load_dataset
from match_model_v2 import EnsembleModel, load_v1_features, v1_predict
from tournament import (QF, R16, R32, SF, _table, derive_groups, edition, label_groups_2026,
                        rank_group, third_place_matchings)

STAGES = ['R32', 'R16', 'QF', 'SF', 'Final', 'Champion']
YEAR_START = pd.Timestamp('2026-06-11')
GROUPS_DONE = pd.Timestamp('2026-06-28')


def rating_state_asof(played, as_of):
    s = RatingState()
    for r in played[played.date < as_of].itertuples():
        s.update(r.home_team, r.away_team, int(r.home_score), int(r.away_score), r.date,
                 bool(r.neutral), r.tournament)
    return s


def actual_r32(gs, ko, groups):
    """Real round-of-32 pairings keyed by FIFA match number (73..88), found by anchoring each
    bracket slot on the group winner / runner-up it contains."""
    std = {}
    for L, teams in groups.items():
        games = [(r.home_team, r.away_team, r.home_score, r.away_score) for r in gs.itertuples()
                 if r.home_team in teams]
        std[L] = rank_group(teams, games, random.Random(0))
    pairs = [(r.home_team, r.away_team) for r in ko.iloc[:16].itertuples()]
    out = {}
    for m, (a, b) in R32.items():
        ta = std[a[1]][int(a[0]) - 1]
        pair = next(p for p in pairs if ta in p)
        out[m] = (ta, pair[1] if pair[0] == ta else pair[0])
    return out, std


def simulate(played, d_all, as_of, n_sims, beta, mode='full', seed=0, rating_asof=None):
    """mode 'full': simulate unplayed group games, derive bracket, play knockouts.
       mode 'knockout_actual': real group results + real R32 pairings, simulate knockouts only.
       rating_asof: date of the ratings used (defaults to as_of)."""
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    gs, ko = edition(d_all, 2026)
    groups = label_groups_2026(derive_groups(gs))
    teams = sorted(t for g in groups.values() for t in g)
    ix = {t: i for i, t in enumerate(teams)}
    N = n_sims
    state = rating_state_asof(played, rating_asof or as_of)
    elo = np.tile(np.array([state.elo(t) for t in teams]), (N, 1))
    rows = np.arange(N)
    counts = {s: np.zeros(len(teams)) for s in STAGES}
    pos_counts = np.zeros((len(teams), 4))

    if mode == 'full':
        train = d_all[d_all.date < as_of]
        age = (as_of - train.date).dt.days.values / 365.25
        gm = GoalModel().fit(train, sample_weight=0.5 ** (age / 15.0))
        ew_gf = np.array([state.team(t).ew_gf for t in teams])
        ew_ga = np.array([state.team(t).ew_ga for t in teams])
        G = np.zeros((N, len(gs), 2), dtype=int)
        for m, r in enumerate(gs.itertuples()):
            ih, ia = ix[r.home_team], ix[r.away_team]
            if r.date < as_of:                            # already played: real score
                G[:, m, 0], G[:, m, 1] = r.home_score, r.away_score
                continue
            ha = 0.0 if r.neutral else 1.0
            dr = elo[:, ih] - elo[:, ia] + HOME_ADV * ha
            one = np.ones(N)
            lh, la = gm.rates_arrays(dr, ha * one, ew_gf[ih] * one, ew_ga[ih] * one,
                                     ew_gf[ia] * one, ew_ga[ia] * one)
            cdf = gm.pmf(lh, la).reshape(N, -1).cumsum(axis=1)
            idx = np.minimum((cdf < rng.random(N)[:, None]).sum(axis=1), cdf.shape[1] - 1)
            gh, ga = idx // (13), idx % 13
            G[:, m, 0], G[:, m, 1] = gh, ga
            we = 1 / (1 + 10 ** (-dr / 400))
            w = np.where(gh > ga, 1.0, np.where(gh < ga, 0.0, 0.5))
            gd = np.abs(gh - ga)
            mult = np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11 + gd) / 8))
            delta = 60.0 * mult * (w - we)
            elo[:, ih] += delta
            elo[:, ia] -= delta
        # ---- group tables and bracket, simulation by simulation
        gm_idx = {L: [m for m, r in enumerate(gs.itertuples()) if r.home_team in groups[L]] for L in groups}
        gm_pairs = {m: (ix[r.home_team], ix[r.away_team]) for m, r in enumerate(gs.itertuples())}
        R32A, R32B = np.zeros((N, 16), int), np.zeros((N, 16), int)
        for n in range(N):
            std, thirds = {}, []
            for L, tl in groups.items():
                tix = [ix[t] for t in tl]
                games = [(gm_pairs[m][0], gm_pairs[m][1], G[n, m, 0], G[n, m, 1]) for m in gm_idx[L]]
                order = rank_group(tix, games, pyrng)
                tbl = _table(tix, games)
                std[L] = order
                for p, t in enumerate(order):
                    pos_counts[t, p] += 1
                t3 = order[2]
                thirds.append((tbl[t3][0], tbl[t3][1], tbl[t3][2], pyrng.random(), L))
            thirds.sort(reverse=True)
            asg = pyrng.choice(third_place_matchings([x[4] for x in thirds[:8]]))
            for m, (a, b) in R32.items():
                R32A[n, m - 73] = std[a[1]][int(a[0]) - 1]
                R32B[n, m - 73] = std[asg[m]][2] if b[0] == '3' else std[b[1]][int(b[0]) - 1]
    else:
        pairs, std = actual_r32(gs, ko, groups)
        R32A = np.tile([ix[pairs[m][0]] for m in range(73, 89)], (N, 1))
        R32B = np.tile([ix[pairs[m][1]] for m in range(73, 89)], (N, 1))
        for L, order in std.items():
            for p, t in enumerate(order):
                pos_counts[ix[t], p] += N

    def play(A, B):
        dr = elo[rows, A] - elo[rows, B]
        p = 1.0 / (1.0 + np.exp(-beta * dr / 100.0))
        a_wins = rng.random(N) < p
        we = 1 / (1 + 10 ** (-dr / 400))
        delta = 60.0 * (a_wins.astype(float) - we)
        elo[rows, A] += delta
        elo[rows, B] -= delta
        return np.where(a_wins, A, B)

    def tally(stage, *arrs):
        for a in arrs:
            counts[stage] += np.bincount(a, minlength=len(teams))

    tally('R32', R32A.reshape(-1), R32B.reshape(-1))
    W = {m: play(R32A[:, m - 73], R32B[:, m - 73]) for m in R32}
    tally('R16', *W.values())
    for stage, rnd in (('QF', R16), ('SF', QF), ('Final', SF)):
        nxt = {m: play(W[a], W[b]) for m, (a, b) in rnd.items()}
        W.update(nxt)
        tally(stage, *nxt.values())
    champ = play(W[101], W[102])
    tally('Champion', champ)

    res = pd.DataFrame({'team': teams})
    for s in STAGES:
        res[f'P_{s}'] = counts[s] / N
    res['P_win_group'] = pos_counts[:, 0] / N
    res['P_top2'] = pos_counts[:, :2].sum(axis=1) / N
    res['group'] = [next(L for L, g in groups.items() if t in g) for t in teams]
    return res


def actual_stages(d_all):
    """Furthest stage actually reached by each 2026 team (index into STAGES, -1 = out in groups)."""
    gs, ko = edition(d_all, 2026)
    ko = ko.reset_index(drop=True)
    reached = {t: -1 for t in set(gs.home_team) | set(gs.away_team)}
    blocks = {0: range(0, 16), 1: range(16, 24), 2: range(24, 28), 3: range(28, 30), 4: [31]}
    for s, idx in blocks.items():
        for i in idx:
            for t in (ko.home_team[i], ko.away_team[i]):
                reached[t] = max(reached[t], s)
    final = ko.iloc[31]
    champ = final.home_team if final.home_score > final.away_score else final.away_team
    reached[champ] = 5
    return reached


def score_stage_probs(res, reached, eligible=None):
    """Brier / log-loss of 'team reaches stage s' forecasts, vs a naive forecast that spreads
    each stage's real number of slots uniformly over the eligible teams."""
    out = []
    elig = res if eligible is None else res[res.team.isin(eligible)]
    for k, s in enumerate(STAGES):
        y = np.array([reached[t] >= k for t in elig.team], dtype=float)
        p = np.clip(elig[f'P_{s}'].values, 1e-3, 1 - 1e-3)
        naive = np.full(len(y), y.mean())
        out.append({'stage': s, 'brier': np.mean((p - y) ** 2), 'brier_naive': np.mean((naive - y) ** 2),
                    'logloss': -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                    'logloss_naive': -np.mean(y * np.log(np.clip(naive, 1e-3, 1)) + (1 - y) * np.log(np.clip(1 - naive, 1e-3, 1)))})
    o = pd.DataFrame(out)
    o['brier_skill'] = np.where(o.brier_naive > 0, 1 - o.brier / o.brier_naive.replace(0, np.nan), np.nan)
    return o


def knockout_match_predictions(d, K, played):
    """Per-match P(advance) for the 31 actual 2026 knockout games from four models."""
    from sklearn.metrics import log_loss
    te = K[K.year == 2026].reset_index(drop=True)
    tr = K[K.year < 2026]
    out = te[['date', 'team1', 'team2', 'y']].copy()
    b_post = fit_symmetric(tr[['d_elo']].values, tr.y.values)
    b_pre = fit_symmetric(tr[['d_elo_pre']].values, tr.y.values)
    out['p_knockout_model_after_groups'] = b_post.predict_proba(te[['d_elo']].values)[:, 1]
    out['p_elo_pre_tournament'] = b_pre.predict_proba(te[['d_elo_pre']].values)[:, 1]
    keyed = d.set_index(['date', 'home_team', 'away_team'])
    rows = keyed.loc[list(zip(te.date, te.team1, te.team2))].reset_index()
    ens = EnsembleModel().fit(d[d.date < YEAR_START], YEAR_START)
    P = ens.predict_proba(rows)
    out['p_v2_match_model'] = P[:, 2] + 0.5 * P[:, 1]
    v1 = load_v1_features()
    if v1 is not None:
        keys = d[['date', 'home_team', 'away_team', 'result']]
        P1 = v1_predict(v1, keys[keys.date < YEAR_START], rows, True)   # original v1 recipe
        out['p_v1_original'] = P1[:, 2] + 0.5 * P1[:, 1]
    rows_m = []
    for c in [c for c in out.columns if c.startswith('p_')]:
        p = out[c].values
        rows_m.append({'model': c[2:], 'n': len(out), 'accuracy': ((p > .5) == out.y).mean(),
                       'log_loss': log_loss(out.y, p), 'brier': np.mean((p - out.y) ** 2)})
    return out, pd.DataFrame(rows_m)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=10000)
    args = ap.parse_args()
    pd.set_option('display.width', 220)

    played = load_played('.')
    d_all, _ = build_features(played)
    d, K = load_dataset()
    tr = K[K.year < 2026]
    beta_post = float(fit_symmetric(tr[['d_elo']].values, tr.y.values).coef_[0][0])
    beta_pre = float(fit_symmetric(tr[['d_elo_pre']].values, tr.y.values).coef_[0][0])
    reached = actual_stages(d_all)
    champ = [t for t, v in reached.items() if v == 5][0]
    print(f'Knockout-model slope (fit on 1986-2022): after-groups {beta_post:.3f}, pre-tournament {beta_pre:.3f}')
    print(f'Actual 2026 champion: {champ}\n')

    runs = {
        'A_pre_tournament': dict(as_of=YEAR_START, mode='full', beta=beta_post),
        'B_after_groups': dict(as_of=GROUPS_DONE, mode='knockout_actual', beta=beta_post),
        'C_after_groups_ignoring_group_stage': dict(as_of=GROUPS_DONE, mode='knockout_actual',
                                                    beta=beta_pre, rating_asof=YEAR_START),
    }
    summary = []
    for name, kw in runs.items():
        res = simulate(played, d_all, n_sims=args.sims, **kw)
        res['actual_stage'] = res.team.map(lambda t: (['Groups'] + STAGES)[reached[t] + 1])
        res = res.sort_values('P_Champion', ascending=False)
        res.round(4).to_csv(f'playoff_probabilities_{name}.csv', index=False)
        elig = None if kw['mode'] == 'full' else [t for t, v in reached.items() if v >= 0]
        sc = score_stage_probs(res, reached, elig)
        c = res.set_index('team')
        print(f'=== {name} ===')
        print(res.head(8)[['team', 'P_R32', 'P_QF', 'P_SF', 'P_Final', 'P_Champion', 'actual_stage']].round(3).to_string(index=False))
        print(f'P(champion = {champ}) = {c.loc[champ, "P_Champion"]:.3f}  (rank {list(res.team).index(champ) + 1}) | '
              f'mean Brier skill over stages vs naive: {sc.brier_skill.mean():.3f}\n')
        summary.append({'run': name, 'p_actual_champion': c.loc[champ, 'P_Champion'],
                        'rank_of_champion': list(res.team).index(champ) + 1,
                        'brier_skill_vs_naive_mean': sc.brier_skill.mean(),
                        **{f'brier_skill_{r.stage}': r.brier_skill for r in sc.itertuples()}})
        # champion log score
    pd.DataFrame(summary).round(4).to_csv('playoff_backtest_summary.csv', index=False)

    preds, mm = knockout_match_predictions(d_all, K, played)
    preds.round(4).to_csv('knockout_predictions_2026.csv', index=False)
    mm.round(4).to_csv('knockout_match_comparison_2026.csv', index=False)
    print('Per-match accuracy on the 31 real 2026 knockout games (3rd-place game excluded):')
    print(mm.round(4).to_string(index=False))
