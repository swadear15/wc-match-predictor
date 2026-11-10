"""
features_v2.py -- improved, leakage-free rating + form features.

Improvements over feature_engineering.py (v1):
  * Elo K-factor depends on match importance (World Cup > continental finals >
    qualifiers/Nations League > other tournaments > friendlies), as in World Football Elo.
  * Home advantage is modelled explicitly (+HOME_ADV Elo for the non-neutral home side),
    instead of being left to the classifier.
  * Opponent-adjusted form ("surprise" = actual - Elo-expected score), Elo trend,
    exponentially weighted attack/defence goal rates, rest days, shrunk head-to-head.
  * One chronological pass builds both the training features AND the live rating state,
    so predicting a future fixture uses exactly the same code path as training.

Everything a row sees is computed from matches strictly BEFORE it.
"""
from collections import deque
import numpy as np
import pandas as pd

INIT_ELO = 1500.0
HOME_ADV = 100.0          # Elo points for a non-neutral home side
K_SCALE = 1.0
EW_ALPHA = 0.08           # smoothing for exponentially-weighted goal rates
PRIOR_GF = 1.3

CONTINENTAL_FINALS = {
    'UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup', 'Gold Cup',
    'Confederations Cup', 'CONCACAF Championship', 'Oceania Nations Cup',
}


def k_factor(tournament: str) -> float:
    if tournament == 'FIFA World Cup':
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if tournament in ('Friendly', 'FIFA Series'):
        return 20.0
    if 'qualification' in tournament or 'Nations League' in tournament:
        return 40.0
    return 30.0


def gd_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


class _Team:
    __slots__ = ('elo', 'hist', 'elo_hist', 'ew_gf', 'ew_ga', 'last', 'n')

    def __init__(self):
        self.elo = INIT_ELO
        self.hist = deque(maxlen=10)       # (points, gf, ga, surprise)
        self.elo_hist = deque(maxlen=10)   # post-match Elo values
        self.ew_gf = PRIOR_GF
        self.ew_ga = PRIOR_GF
        self.last = None
        self.n = 0


def _form(team: _Team, window: int):
    h = list(team.hist)[-window:]
    if not h:
        return 1.2, PRIOR_GF, PRIOR_GF, 0.0
    a = np.asarray(h, dtype=float)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()


class RatingState:
    """Mutable rating/form state. Call .features() BEFORE .update() for a match."""

    def __init__(self, k_scale=K_SCALE, home_adv=HOME_ADV):
        self.teams = {}
        self.h2h = {}
        self.k_scale = k_scale
        self.home_adv = home_adv

    def team(self, name) -> _Team:
        t = self.teams.get(name)
        if t is None:
            t = self.teams[name] = _Team()
        return t

    def elo(self, name) -> float:
        return self.team(name).elo

    # ------------------------------------------------------------------ features
    def features(self, home, away, date, neutral, tournament) -> dict:
        th, ta = self.team(home), self.team(away)
        ha = 0.0 if neutral else 1.0
        elo_diff = th.elo - ta.elo
        elo_diff_adj = elo_diff + self.home_adv * ha
        f = {
            'elo_h': th.elo, 'elo_a': ta.elo, 'elo_diff': elo_diff,
            'home_flag': ha, 'elo_diff_adj': elo_diff_adj,
            'abs_elo_diff_adj': abs(elo_diff_adj),
            'we_h': 1.0 / (1.0 + 10.0 ** (-elo_diff_adj / 400.0)),
            'k_importance': k_factor(tournament),
        }
        for side, t in (('h', th), ('a', ta)):
            p5, gf5, ga5, s5 = _form(t, 5)
            p10, gf10, ga10, s10 = _form(t, 10)
            f[f'ppg5_{side}'], f[f'gf5_{side}'], f[f'ga5_{side}'] = p5, gf5, ga5
            f[f'ppg10_{side}'], f[f'gf10_{side}'], f[f'ga10_{side}'] = p10, gf10, ga10
            f[f'surprise10_{side}'] = s10
            f[f'ew_gf_{side}'], f[f'ew_ga_{side}'] = t.ew_gf, t.ew_ga
            f[f'elo_trend_{side}'] = (t.elo - t.elo_hist[0]) if t.elo_hist else 0.0
            f[f'rest_{side}'] = min((date - t.last).days, 90) if t.last is not None else 30
            f[f'n_{side}'] = min(t.n, 200)
        key = (home, away) if home < away else (away, home)
        rec = self.h2h.get(key)
        if rec:
            n = rec['n']
            f['h2h_n'] = n
            f['h2h_edge'] = (rec.get(home, 0) - rec.get(away, 0)) / (n + 3.0)
        else:
            f['h2h_n'], f['h2h_edge'] = 0, 0.0
        return f

    # -------------------------------------------------------------------- update
    def update(self, home, away, hs, as_, date, neutral, tournament) -> float:
        """Apply a played match. Returns the pre-match expected score of the home side."""
        th, ta = self.team(home), self.team(away)
        ha = 0.0 if neutral else 1.0
        dr = th.elo - ta.elo + self.home_adv * ha
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_h = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        delta = k_factor(tournament) * self.k_scale * gd_multiplier(abs(hs - as_)) * (w_h - we_h)
        pts_h = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_a = 3 if hs < as_ else (1 if hs == as_ else 0)
        sur = w_h - we_h
        th.hist.append((pts_h, hs, as_, sur))
        ta.hist.append((pts_a, as_, hs, -sur))
        th.elo += delta
        ta.elo -= delta
        th.elo_hist.append(th.elo)
        ta.elo_hist.append(ta.elo)
        th.ew_gf += EW_ALPHA * (hs - th.ew_gf)
        th.ew_ga += EW_ALPHA * (as_ - th.ew_ga)
        ta.ew_gf += EW_ALPHA * (as_ - ta.ew_gf)
        ta.ew_ga += EW_ALPHA * (hs - ta.ew_ga)
        th.last = ta.last = date
        th.n += 1
        ta.n += 1
        key = (home, away) if home < away else (away, home)
        rec = self.h2h.setdefault(key, {'n': 0})
        rec['n'] += 1
        if hs > as_:
            rec[home] = rec.get(home, 0) + 1
        elif hs < as_:
            rec[away] = rec.get(away, 0) + 1
        return we_h


FEATURE_COLS = [
    'elo_h', 'elo_a', 'elo_diff_adj', 'abs_elo_diff_adj', 'home_flag', 'k_importance',
    'ppg5_h', 'ppg5_a', 'ppg10_h', 'ppg10_a',
    'gf5_h', 'gf5_a', 'ga5_h', 'ga5_a', 'gf10_h', 'gf10_a', 'ga10_h', 'ga10_a',
    'surprise10_h', 'surprise10_a', 'ew_gf_h', 'ew_gf_a', 'ew_ga_h', 'ew_ga_a',
    'elo_trend_h', 'elo_trend_a', 'rest_h', 'rest_a', 'n_h', 'n_a', 'h2h_n', 'h2h_edge',
]


def build_features(played: pd.DataFrame, k_scale=K_SCALE, home_adv=HOME_ADV):
    """played: chronologically sorted frame with date, home_team, away_team, home_score,
    away_score, tournament, neutral. Returns (featured_frame, final RatingState)."""
    state = RatingState(k_scale=k_scale, home_adv=home_adv)
    rows = []
    for r in played.itertuples(index=False):
        neutral = bool(r.neutral)
        f = state.features(r.home_team, r.away_team, r.date, neutral, r.tournament)
        state.update(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                     r.date, neutral, r.tournament)
        rows.append(f)
    feats = pd.DataFrame(rows, index=played.index)
    return pd.concat([played, feats], axis=1), state


def load_played(path_dir='.'):
    """Load results.csv -> played matches (with result label), chronologically sorted."""
    df = pd.read_csv(f'{path_dir}/results.csv', parse_dates=['date'])
    df = df[df.home_score.notna()].copy()
    df['home_score'] = df.home_score.astype(int)
    df['away_score'] = df.away_score.astype(int)
    df['neutral'] = df.neutral.astype(str).str.upper().eq('TRUE')
    df['result'] = np.where(df.home_score > df.away_score, 'H',
                            np.where(df.home_score < df.away_score, 'A', 'D'))
    return df.sort_values('date', kind='stable').reset_index(drop=True)
