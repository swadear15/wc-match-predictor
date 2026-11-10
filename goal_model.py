"""
goal_model.py -- Poisson goal model (with Dixon-Coles low-score correction).

Why: a win/draw/loss classifier cannot produce scorelines, but group tables need goal
difference and goals scored for tie-breaks. This model predicts expected goals for each
side, builds a joint scoreline distribution, and gives W/D/L probabilities that are
consistent with the simulated scorelines.
"""
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAXG = 12  # scorelines 0..MAXG goals per side

GOAL_FEATS = ['d_elo', 'ha', 'own_gf', 'opp_ga', 'own_ga', 'opp_gf']


def _design(elo_diff_adj, home_flag, ew_gf_h, ew_ga_h, ew_gf_a, ew_ga_a):
    """Stacked (home-perspective, away-perspective) design matrices."""
    xh = np.column_stack([elo_diff_adj / 100.0, home_flag, ew_gf_h, ew_ga_a, ew_ga_h, ew_gf_a])
    xa = np.column_stack([-elo_diff_adj / 100.0, -home_flag, ew_gf_a, ew_ga_h, ew_ga_a, ew_gf_h])
    return xh, xa


def design_from_frame(df):
    return _design(df.elo_diff_adj.values, df.home_flag.values, df.ew_gf_h.values,
                   df.ew_ga_h.values, df.ew_gf_a.values, df.ew_ga_a.values)


class GoalModel:
    def __init__(self, alpha=1e-3):
        self.reg = PoissonRegressor(alpha=alpha, max_iter=2000)
        self.rho = 0.0

    def fit(self, df, sample_weight=None):
        xh, xa = design_from_frame(df)
        X = np.vstack([xh, xa])
        y = np.concatenate([df.home_score.values, df.away_score.values])
        w = None if sample_weight is None else np.concatenate([sample_weight, sample_weight])
        self.reg.fit(X, y, sample_weight=w)
        # fit the Dixon-Coles rho by maximum likelihood on (a sample of) the training set
        sub = df if len(df) <= 15000 else df.iloc[-15000:]
        lh, la = self.rates_frame(sub)
        best, best_ll = 0.0, -np.inf
        for rho in np.linspace(-0.25, 0.10, 36):
            p = self._pmf(lh, la, rho)
            idx = np.arange(len(sub))
            hs = np.clip(sub.home_score.values, 0, MAXG)
            as_ = np.clip(sub.away_score.values, 0, MAXG)
            ll = np.log(np.maximum(p[idx, hs, as_], 1e-12)).sum()
            if ll > best_ll:
                best, best_ll = rho, ll
        self.rho = best
        return self

    # ---------------------------------------------------------------- rates / pmf
    def rates_arrays(self, elo_diff_adj, home_flag, ew_gf_h, ew_ga_h, ew_gf_a, ew_ga_a):
        xh, xa = _design(np.asarray(elo_diff_adj, float), np.asarray(home_flag, float),
                         np.asarray(ew_gf_h, float), np.asarray(ew_ga_h, float),
                         np.asarray(ew_gf_a, float), np.asarray(ew_ga_a, float))
        c, b = self.reg.coef_, self.reg.intercept_
        return np.exp(xh @ c + b), np.exp(xa @ c + b)

    def rates_frame(self, df):
        return self.rates_arrays(df.elo_diff_adj.values, df.home_flag.values, df.ew_gf_h.values,
                                 df.ew_ga_h.values, df.ew_gf_a.values, df.ew_ga_a.values)

    def _pmf(self, lh, la, rho):
        lh = np.atleast_1d(lh)[:, None]
        la = np.atleast_1d(la)[:, None]
        g = np.arange(MAXG + 1)[None, :]
        ph = poisson.pmf(g, lh)            # (n, G)
        pa = poisson.pmf(g, la)
        p = ph[:, :, None] * pa[:, None, :]  # (n, G, G)
        lh_, la_ = lh[:, 0], la[:, 0]
        p[:, 0, 0] *= np.maximum(1 - lh_ * la_ * rho, 1e-6)
        p[:, 0, 1] *= np.maximum(1 + lh_ * rho, 1e-6)
        p[:, 1, 0] *= np.maximum(1 + la_ * rho, 1e-6)
        p[:, 1, 1] *= np.maximum(1 - rho, 1e-6)
        return p / p.sum(axis=(1, 2), keepdims=True)

    def pmf(self, lh, la):
        return self._pmf(lh, la, self.rho)

    @staticmethod
    def wdl(p):
        """p: (n,G,G) -> columns [P(away win), P(draw), P(home win)]"""
        n = p.shape[0]
        i, j = np.indices(p.shape[1:])
        pa = p[:, i < j].sum(axis=1)
        pd_ = p[:, i == j].sum(axis=1)
        ph = p[:, i > j].sum(axis=1)
        return np.column_stack([pa, pd_, ph])

    def predict_proba_frame(self, df):
        lh, la = self.rates_frame(df)
        return self.wdl(self.pmf(lh, la))   # order A, D, H (alphabetical, matches sklearn)
