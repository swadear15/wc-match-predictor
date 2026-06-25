import pandas as pd
import numpy as np
import joblib

# Reload everything needed
hgb = joblib.load('model_hgb.pkl')
logreg = joblib.load('model_logreg.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

matches_featured = pd.read_csv('matches_featured.csv', parse_dates=['date'])
fixtures = pd.read_csv('fixtures_upcoming.csv', parse_dates=['date'])

feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

# We need CURRENT (as of latest match) Elo + form + h2h for every team appearing
# in the upcoming fixtures. Reconstruct each team's latest known state from
# matches_featured (which already has pre-match features computed chronologically;
# we want the POST-match state of each team's most recent game = "pre" state of their NEXT game).

# Easiest robust approach: recompute Elo/form state at the end of matches_featured
# using same logic as feature_engineering.py, then carry forward into fixtures.

elo = {}
INIT_ELO = 1500

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def goal_diff_multiplier(gd):
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8

K = 20
matches_sorted = matches_featured.sort_values('date')
for _, row in matches_sorted.iterrows():
    h, a = row['home_team'], row['away_team']
    r_h, r_a = elo.get(h, INIT_ELO), elo.get(a, INIT_ELO)
    if row['result'] == 'H':
        s_h, s_a = 1, 0
    elif row['result'] == 'A':
        s_h, s_a = 0, 1
    else:
        s_h, s_a = 0.5, 0.5
    exp_h = expected_score(r_h, r_a)
    exp_a = 1 - exp_h
    gd = abs(row['home_score'] - row['away_score'])
    mult = goal_diff_multiplier(gd)
    elo[h] = r_h + K * mult * (s_h - exp_h)
    elo[a] = r_a + K * mult * (s_a - exp_a)

# Rolling form: take each team's last known rolling stats (post most-recent match)
team_long_home = matches_sorted[['date','home_team','home_score','away_score','result']].rename(
    columns={'home_team':'team','home_score':'goals_for','away_score':'goals_against'})
team_long_home['points'] = team_long_home['result'].map({'H':3,'D':1,'A':0})

team_long_away = matches_sorted[['date','away_team','home_score','away_score','result']].rename(
    columns={'away_team':'team','away_score':'goals_for','home_score':'goals_against'})
team_long_away['points'] = team_long_away['result'].map({'A':3,'D':1,'H':0})

team_long = pd.concat([team_long_home, team_long_away], ignore_index=True).sort_values(['team','date'])

latest_form = {}
for window in [5, 10]:
    grp = team_long.groupby('team')
    for team, sub in grp:
        sub = sub.sort_values('date')
        latest_form.setdefault(team, {})
        latest_form[team][f'form_pts_{window}'] = sub['points'].tail(window).mean()
        latest_form[team][f'goals_for_avg_{window}'] = sub['goals_for'].tail(window).mean()
        latest_form[team][f'goals_against_avg_{window}'] = sub['goals_against'].tail(window).mean()

matches_played_count = team_long.groupby('team').size().to_dict()

# H2H: build lookup of total meetings + home-perspective win rate from full history
h2h_record = {}  # key: frozenset({teamA,teamB}) -> {'wins': {team: n}, 'draws': n, 'total': n}
for _, row in matches_sorted.iterrows():
    key = tuple(sorted([row['home_team'], row['away_team']]))
    rec = h2h_record.setdefault(key, {'wins': {}, 'draws': 0, 'total': 0})
    rec['total'] += 1
    if row['result'] == 'H':
        rec['wins'][row['home_team']] = rec['wins'].get(row['home_team'], 0) + 1
    elif row['result'] == 'A':
        rec['wins'][row['away_team']] = rec['wins'].get(row['away_team'], 0) + 1
    else:
        rec['draws'] += 1

def get_h2h(home, away):
    key = tuple(sorted([home, away]))
    rec = h2h_record.get(key)
    if not rec or rec['total'] == 0:
        return 0, np.nan
    win_rate = rec['wins'].get(home, 0) / rec['total']
    return rec['total'], win_rate

# Build feature rows for upcoming fixtures
rows = []
for _, fx in fixtures.iterrows():
    h, a = fx['home_team'], fx['away_team']
    r_h, r_a = elo.get(h, INIT_ELO), elo.get(a, INIT_ELO)
    h2h_count, h2h_rate = get_h2h(h, a)
    hf = latest_form.get(h, {})
    af = latest_form.get(a, {})
    row = {
        'date': fx['date'], 'home_team': h, 'away_team': a,
        'elo_diff': r_h - r_a, 'home_elo_pre': r_h, 'away_elo_pre': r_a,
        'neutral': int(fx['neutral']),
        'home_form_pts_5': hf.get('form_pts_5', np.nan),
        'away_form_pts_5': af.get('form_pts_5', np.nan),
        'home_form_pts_10': hf.get('form_pts_10', np.nan),
        'away_form_pts_10': af.get('form_pts_10', np.nan),
        'home_goals_for_avg_5': hf.get('goals_for_avg_5', np.nan),
        'away_goals_for_avg_5': af.get('goals_for_avg_5', np.nan),
        'home_goals_against_avg_5': hf.get('goals_against_avg_5', np.nan),
        'away_goals_against_avg_5': af.get('goals_against_avg_5', np.nan),
        'home_matches_played_before': matches_played_count.get(h, 0),
        'away_matches_played_before': matches_played_count.get(a, 0),
        'h2h_matches_count': h2h_count,
        'h2h_home_win_rate': h2h_rate,
    }
    rows.append(row)

pred_df = pd.DataFrame(rows)
X_pred = pred_df[feature_cols]

# HGB predictions (native NaN handling)
hgb_proba = hgb.predict_proba(X_pred)
hgb_pred = hgb.predict(X_pred)

# LogReg predictions (needs impute + scale)
X_pred_imp = imputer.transform(X_pred)
X_pred_scaled = scaler.transform(X_pred_imp)
logreg_proba = logreg.predict_proba(X_pred_scaled)
logreg_pred = logreg.predict(X_pred_scaled)

classes = hgb.classes_  # ['A','D','H'] alphabetical
out = pred_df[['date','home_team','away_team']].copy()
out['hgb_prediction'] = hgb_pred
for i, c in enumerate(classes):
    out[f'hgb_prob_{c}'] = hgb_proba[:, i].round(3)
out['logreg_prediction'] = logreg_pred
for i, c in enumerate(classes):
    out[f'logreg_prob_{c}'] = logreg_proba[:, i].round(3)

out.to_csv('fixture_predictions.csv', index=False)
print(out.to_string(index=False))
print()
print(f"Saved {len(out)} predictions to fixture_predictions.csv")
