"""
knockout_model.py -- the "group stage -> playoffs" model.

Question: once the group stage is over, what does it tell us about who wins the knockouts?
We fit P(team 1 advances) on every World Cup knockout match since 1986 (3rd-place games
excluded) and test candidate predictors:

    d_elo_pre      Elo gap BEFORE the tournament                (ignores the group stage)
    d_elo          Elo gap going into the knockout match        (Elo is updated after every
                                                                 group game -> group results
                                                                 flow into team strength)
    + group-stage stats: over-performance vs Elo expectation, goal diff / game, points /
      game, goals / game, "won the group"

Logistic regression with NO intercept, trained on both orientations of each match (A vs B and
B vs A), so the model is exactly symmetric in the two teams. Evaluation is
leave-one-World-Cup-out, plus a pure hold-out: fit on 1986-2022, predict 2026.

    python knockout_model.py            # evaluate + write knockout_model_cv.csv, fit final model
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from features_v2 import build_features, load_played
from goal_model import GoalModel
from tournament import knockout_dataset, shootout_map

warnings.filterwarnings('ignore')

FEATURE_SETS = {
    'pre-tournament Elo only': ['d_elo_pre'],
    'Elo after group stage': ['d_elo'],
    'Elo + group over-performance': ['d_elo', 'd_overperf'],
    'Elo + group goal diff/game': ['d_elo', 'd_gd_pg'],
    'Elo + group points/game': ['d_elo', 'd_ppg'],
    'Elo + group goals/game': ['d_elo', 'd_gf_pg'],
    'Elo + won group': ['d_elo', 'd_won_group'],
    'pre Elo + all group stats': ['d_elo_pre', 'd_overperf', 'd_gd_pg', 'd_ppg', 'd_won_group'],
}
DEFAULT_COLS = ['d_elo']
C_REG = 1.0


def fit_symmetric(X, y, C=C_REG):
    Xm, ym = np.vstack([X, -X]), np.concatenate([y, 1 - y])
    return LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(Xm, ym)


def advance_prob(coef, X):
    return 1.0 / (1.0 + np.exp(-(X @ coef)))


def loeo(K, cols, years):
    """Leave-one-World-Cup-out predictions for the given editions."""
    pred = pd.Series(index=K.index, dtype=float)
    for yr in years:
        tr, te = K[K.year != yr], K[K.year == yr]
        m = fit_symmetric(tr[cols].values, tr.y.values)
        pred.loc[te.index] = m.predict_proba(te[cols].values)[:, 1]
    return pred


def match_model_advance_prob(d, K):
    """Baseline: Poisson match model (trained on data before each edition) ->
    P(advance) = P(win in 90) + 0.5 * P(draw)."""
    out = pd.Series(index=K.index, dtype=float)
    keyed = d.set_index(['date', 'home_team', 'away_team'])
    for yr in sorted(K.year.unique()):
        if yr < 2010:
            continue
        start = d[(d.tournament == 'FIFA World Cup') & (d.date.dt.year == yr)].date.min()
        g = GoalModel().fit(d[d.date < start])
        sub = K[K.year == yr]
        rows = keyed.loc[list(zip(sub.date, sub.team1, sub.team2))].reset_index()
        P = g.predict_proba_frame(rows)
        out.loc[sub.index] = P[:, 2] + 0.5 * P[:, 1]
    return out


def evaluate(K, d):
    train_years = sorted(K[K.year < 2026].year.unique())
    rows = []
    for name, cols in FEATURE_SETS.items():
        p = loeo(K[K.year < 2026], cols, train_years)
        sub = K[K.year < 2026]
        m = fit_symmetric(sub[cols].values, sub.y.values)
        te = K[K.year == 2026]
        pt = m.predict_proba(te[cols].values)[:, 1]
        rows.append({
            'feature_set': name,
            'LOEO_logloss_1986-2022': log_loss(sub.y, p), 'LOEO_accuracy_1986-2022': ((p > .5) == sub.y).mean(),
            'holdout_2026_logloss': log_loss(te.y, pt), 'holdout_2026_accuracy': ((pt > .5) == te.y).mean(),
            'coefficients': dict(zip(cols, np.round(m.coef_[0], 3))),
        })
    return pd.DataFrame(rows)


def pick_feature_set(cv, tol=0.01):
    """Simplest feature set whose leave-one-edition-out log-loss is within `tol` of the best."""
    cv = cv.assign(n_feats=cv.coefficients.map(len)).sort_values('LOEO_logloss_1986-2022')
    best = cv['LOEO_logloss_1986-2022'].iloc[0]
    ok = cv[cv['LOEO_logloss_1986-2022'] <= best + tol].sort_values(['n_feats', 'LOEO_logloss_1986-2022'])
    return list(ok.iloc[0].coefficients.keys()), ok.iloc[0].feature_set


def load_dataset():
    d, _ = build_features(load_played('.'))
    K = knockout_dataset(d, list(range(1986, 2023, 4)) + [2026], shootout_map())
    return d, K.reset_index(drop=True)


if __name__ == '__main__':
    d, K = load_dataset()
    print(f'{len(K)} knockout matches ({(K.year < 2026).sum()} in 1986-2022, {(K.year == 2026).sum()} in 2026)')
    cv = evaluate(K, d)
    cols, chosen = pick_feature_set(cv)
    pd.set_option('display.width', 250, 'display.max_colwidth', 80)
    print(cv.round(4).to_string(index=False))
    print(f'\ncoin-flip log-loss = {np.log(2):.4f}.   Chosen (simplest within 0.01 of best LOEO): {chosen} {cols}')

    # does a match model do better than the dedicated knockout model? (2010-2026 editions)
    K2 = K[K.year >= 2010]
    p_ko = loeo(K, cols, sorted(K.year.unique())).loc[K2.index]
    p_mm = match_model_advance_prob(d, K).loc[K2.index]
    print('\nSame 2010-2026 knockout matches (n=%d):' % len(K2))
    for nm, p in (('knockout model (Elo after group stage)', p_ko), ('Poisson match model, P(win)+0.5*P(draw)', p_mm)):
        print(f'  {nm:45s} log-loss={log_loss(K2.y, p):.4f}  accuracy={((p > .5) == K2.y).mean():.3f}')
    cv.assign(coefficients=cv.coefficients.astype(str)).to_csv('knockout_model_cv.csv', index=False)

    # final production model: fit on every knockout match 1986-2026
    m = fit_symmetric(K[cols].values, K.y.values)
    json.dump({'features': cols, 'coef': [float(c) for c in m.coef_[0]], 'n_matches': int(len(K)),
               'note': 'P(team1 advances) = sigmoid(sum(coef*feature)); d_elo = Elo gap/100'},
              open('knockout_model.json', 'w'), indent=2)
    print('saved knockout_model.json', dict(zip(cols, m.coef_[0].round(3))))
