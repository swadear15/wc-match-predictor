"""
match_model_v2.py -- improved win/draw/loss model + honest time-ordered backtest.

    python match_model_v2.py backtest   # expanding-window backtest vs the v1 model
    python match_model_v2.py train      # fit on all data, save model_v2.pkl

Model = average of three probabilistic models (all see only pre-match information):
    1. standardised multinomial logistic regression on the v2 features
    2. gradient boosting (shallow, regularised, recency-weighted, post-1990 data)
    3. Poisson goal model with Dixon-Coles correction (see goal_model.py)

Differences from v1 (finalize_model.py):
    * NO class re-weighting. Re-weighting forces the model to over-predict draws, which
      wrecks probability calibration and lowers accuracy. Probabilities are kept honest;
      the label is simply the most likely outcome.
    * Evaluation is time-ordered (train on the past, test on the future) instead of
      shuffled 5-fold CV plus a 20-match validation set.
"""
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_v2 import FEATURE_COLS, build_features, load_played
from goal_model import GoalModel

warnings.filterwarnings('ignore')
CLASSES = np.array(['A', 'D', 'H'])
HALF_LIFE_YEARS = 15.0
HGB_SINCE = '1990-01-01'


def recency_weights(dates, cutoff, half_life=HALF_LIFE_YEARS):
    age_years = (pd.Timestamp(cutoff) - dates).dt.days.values / 365.25
    return 0.5 ** (age_years / half_life)


class EnsembleModel:
    def __init__(self):
        self.parts = {}

    def fit(self, train, cutoff):
        X = FEATURE_COLS
        w = recency_weights(train.date, cutoff)
        self.parts['logit'] = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(train[X], train.result)
        t = train[train.date >= HGB_SINCE]
        self.parts['hgb'] = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_depth=4, min_samples_leaf=80,
            l2_regularization=1.0, random_state=0
        ).fit(t[X], t.result, sample_weight=recency_weights(t.date, cutoff))
        self.parts['goals'] = GoalModel().fit(train, sample_weight=w)
        return self

    def component_probas(self, df):
        X = FEATURE_COLS
        return {
            'v2_logit': self.parts['logit'].predict_proba(df[X]),
            'v2_hgb': self.parts['hgb'].predict_proba(df[X]),
            'v2_poisson': self.parts['goals'].predict_proba_frame(df),
        }

    def predict_proba(self, df):
        comp = self.component_probas(df)
        return np.mean(list(comp.values()), axis=0)   # columns A, D, H


# --------------------------------------------------------------------------- metrics
def rps(y, P):
    o = np.column_stack([(y == 'A'), (y == 'A') | (y == 'D')]).astype(float)
    c = np.column_stack([P[:, 0], P[:, 0] + P[:, 1]])
    return float(((c - o) ** 2).sum(axis=1).mean() / 2)


def metrics(y, P):
    y = np.asarray(y)
    pred = CLASSES[P.argmax(1)]
    return {
        'n': len(y), 'accuracy': accuracy_score(y, pred),
        'log_loss': log_loss(y, P, labels=list(CLASSES)), 'rps': rps(y, P),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'draws_predicted_%': 100 * (pred == 'D').mean(),
    }


# ------------------------------------------------------------------------- v1 model
V1_BASE = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral', 'home_form_pts_5', 'away_form_pts_5',
           'home_form_pts_10', 'away_form_pts_10', 'home_goals_for_avg_5', 'away_goals_for_avg_5',
           'home_goals_against_avg_5', 'away_goals_against_avg_5', 'home_matches_played_before',
           'away_matches_played_before', 'h2h_matches_count', 'h2h_home_win_rate']


def load_v1_features(path='matches_featured.csv'):
    try:
        z = pd.read_csv(path, parse_dates=['date']).sort_values('date', kind='stable')
    except FileNotFoundError:
        return None
    z = z.drop_duplicates(['date', 'home_team', 'away_team']).copy()
    z['abs_elo_diff'] = z.elo_diff.abs()
    z['elo_close'] = (z.abs_elo_diff < 50).astype(int)
    z['form_diff'] = z.home_form_pts_5 - z.away_form_pts_5
    z['abs_form_diff'] = z.form_diff.abs()
    z['combined_goal_avg'] = (z.home_goals_for_avg_5.fillna(1) + z.away_goals_for_avg_5.fillna(1)) / 2
    z['low_scoring_signal'] = (z.combined_goal_avg < 1.2).astype(int)
    z['neutral'] = z.neutral.astype(int)
    cols = V1_BASE + ['abs_elo_diff', 'elo_close', 'form_diff', 'abs_form_diff',
                      'combined_goal_avg', 'low_scoring_signal']
    return z[['date', 'home_team', 'away_team'] + cols], cols


def v1_predict(v1, train_keys, test_keys, class_weighted=True):
    frame, cols = v1
    tr = frame.merge(train_keys, on=['date', 'home_team', 'away_team'])
    te = test_keys[['date', 'home_team', 'away_team']].merge(frame, on=['date', 'home_team', 'away_team'], how='left')
    w = None
    if class_weighted:     # exactly what finalize_model.py does
        cc = tr.result.value_counts()
        w = tr.result.map({c: len(tr) / (3 * cc[c]) for c in cc.index}).values
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    m.fit(tr[cols], tr.result, sample_weight=w)
    return m.predict_proba(te[cols])


# ----------------------------------------------------------------------- backtest
WC_EDITIONS = {2010: '2010-06-11', 2014: '2014-06-12', 2018: '2018-06-14',
               2022: '2022-11-20', 2026: '2026-06-11'}


def run_backtest(out_csv='backtest_match_models.csv'):
    played = load_played('.')
    d, _ = build_features(played)
    v1 = load_v1_features()
    if v1 is None:
        print('(matches_featured.csv not found -> run feature_engineering.py to include the v1 rows)')
    keys = d[['date', 'home_team', 'away_team', 'result']]

    tests = []
    for yr, start in WC_EDITIONS.items():
        s = pd.Timestamp(start)
        m = (d.tournament == 'FIFA World Cup') & (d.date >= s) & (d.date < s + pd.Timedelta(days=45))
        tests.append((f'World Cup {yr}', s, m))
    tests.append(('Competitive matches 2023-01..2026-06 (excl. friendlies)', pd.Timestamp('2023-01-01'),
                  (d.date >= '2023-01-01') & (d.date < '2026-06-11') & (d.tournament != 'Friendly')))

    rows, pooled = [], {}
    for name, cutoff, mask in tests:
        train, test = d[d.date < cutoff], d[mask]
        print(f'{name}: train {len(train)}, test {len(test)}', flush=True)
        ens = EnsembleModel().fit(train, cutoff)
        probs = ens.component_probas(test)
        probs['v2_ensemble'] = ens.predict_proba(test)
        # reference: Elo-only logistic (no form, no goals)
        el = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)).fit(
            train[['elo_diff_adj', 'abs_elo_diff_adj']], train.result)
        probs['elo_only_logit'] = el.predict_proba(test[['elo_diff_adj', 'abs_elo_diff_adj']])
        if v1 is not None:
            probs['v1_class_weighted (original)'] = v1_predict(v1, keys[keys.date < cutoff], test, True)
            probs['v1_features_unweighted'] = v1_predict(v1, keys[keys.date < cutoff], test, False)
        for mname, P in probs.items():
            rows.append({'test_set': name, 'model': mname, **metrics(test.result.values, P)})
            if name.startswith('World Cup'):
                pooled.setdefault(mname, []).append((test.result.values, P))
    for mname, parts in pooled.items():
        y = np.concatenate([p[0] for p in parts])
        P = np.vstack([p[1] for p in parts])
        rows.append({'test_set': 'World Cups 2010-2026 pooled', 'model': mname, **metrics(y, P)})
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    show = res[res.test_set.isin(['World Cups 2010-2026 pooled',
                                  'Competitive matches 2023-01..2026-06 (excl. friendlies)'])]
    pd.set_option('display.width', 200)
    print(show.round(4).to_string(index=False))
    return res


def train_final(path='model_v2.pkl'):
    played = load_played('.')
    d, state = build_features(played)
    cutoff = d.date.max() + pd.Timedelta(days=1)
    ens = EnsembleModel().fit(d, cutoff)
    joblib.dump({'ensemble': ens, 'cutoff': cutoff}, path)
    print(f'trained on {len(d)} matches through {d.date.max().date()} -> {path}')
    return ens, state


def predict_group_stage_2026(out_csv='group_stage_predictions_2026.csv'):
    """All 72 group games of the 2026 World Cup, predicted from the eve of the tournament
    (model fit only on matches before 2026-06-11), next to the real results."""
    cutoff = pd.Timestamp('2026-06-11')
    played = load_played('.')
    d, _ = build_features(played)
    gs = d[(d.tournament == 'FIFA World Cup') & (d.date >= cutoff)].sort_values('date', kind='stable').iloc[:72]
    ens = EnsembleModel().fit(d[d.date < cutoff], cutoff)
    P = ens.predict_proba(gs)
    out = gs[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']].copy()
    out['v2_prob_away_win'], out['v2_prob_draw'], out['v2_prob_home_win'] = P.T.round(3)
    out['v2_pick'] = CLASSES[P.argmax(1)]
    v1 = load_v1_features()
    P1 = None
    if v1 is not None:
        keys = d[['date', 'home_team', 'away_team', 'result']]
        P1 = v1_predict(v1, keys[keys.date < cutoff], gs, True)
        out['v1_pick'] = CLASSES[P1.argmax(1)]
    out['in_original_52_unplayed_fixtures'] = out.date >= '2026-06-17'
    out.to_csv(out_csv, index=False)
    for label, m in (('all 72 group games', np.ones(len(out), bool)),
                     ('the 52 games that were unplayed in the original data snapshot', out.in_original_52_unplayed_fixtures.values)):
        print(f'-- {label}')
        print('   v2 ensemble:', {k: round(v, 4) for k, v in metrics(out.result.values[m], P[m]).items()})
        if P1 is not None:
            print('   v1 original:', {k: round(v, 4) for k, v in metrics(out.result.values[m], P1[m]).items()})


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'backtest'
    {'backtest': run_backtest, 'train': train_final, 'groups2026': predict_group_stage_2026}[cmd]()
