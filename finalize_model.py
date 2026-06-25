import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

train = pd.read_csv('train_set.csv', parse_dates=['date'])
matches_featured = pd.read_csv('matches_featured.csv', parse_dates=['date'])
fixtures = pd.read_csv('fixtures_upcoming.csv', parse_dates=['date'])

base_feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

def add_draw_features(df):
    df['abs_elo_diff'] = df['elo_diff'].abs()
    df['elo_close'] = (df['abs_elo_diff'] < 50).astype(int)
    df['form_diff'] = df['home_form_pts_5'] - df['away_form_pts_5']
    df['abs_form_diff'] = df['form_diff'].abs()
    df['combined_goal_avg'] = (df['home_goals_for_avg_5'].fillna(1) + df['away_goals_for_avg_5'].fillna(1)) / 2
    df['low_scoring_signal'] = (df['combined_goal_avg'] < 1.2).astype(int)
    return df

train = add_draw_features(train)
feature_cols = base_feature_cols + ['abs_elo_diff', 'elo_close', 'form_diff', 'abs_form_diff',
                                      'combined_goal_avg', 'low_scoring_signal']

X_train = train[feature_cols].copy()
X_train['neutral'] = X_train['neutral'].astype(int)
y_train = train['result']

class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {c: total / (len(class_counts) * class_counts[c]) for c in class_counts.index}
sample_weight = y_train.map(class_weights).values

print(f"FINAL MODEL: class-weighted HistGradientBoosting, 3-class")
print(f"Class weights: {class_weights}")
print(f"Training on {len(X_train)} matches...")

final_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
final_model.fit(X_train, y_train, sample_weight=sample_weight)

joblib.dump(final_model, 'model_final.pkl')
print("Saved as model_final.pkl\n")

# =========================================================
# Regenerate fixture predictions with the final model
# =========================================================
elo = {}
INIT_ELO = 1500
def expected_score(r_a, r_b): return 1 / (1 + 10 ** ((r_b - r_a) / 400))
def goal_diff_multiplier(gd):
    if gd <= 1: return 1.0
    elif gd == 2: return 1.5
    else: return (11 + gd) / 8

K = 20
matches_sorted = matches_featured.sort_values('date')
for _, row in matches_sorted.iterrows():
    h, a = row['home_team'], row['away_team']
    r_h, r_a = elo.get(h, INIT_ELO), elo.get(a, INIT_ELO)
    if row['result'] == 'H': s_h, s_a = 1, 0
    elif row['result'] == 'A': s_h, s_a = 0, 1
    else: s_h, s_a = 0.5, 0.5
    exp_h = expected_score(r_h, r_a)
    gd = abs(row['home_score'] - row['away_score'])
    mult = goal_diff_multiplier(gd)
    elo[h] = r_h + K * mult * (s_h - exp_h)
    elo[a] = r_a + K * mult * ((1-exp_h) - s_a if False else (s_a - (1-exp_h)))

team_long_home = matches_sorted[['date','home_team','home_score','away_score','result']].rename(
    columns={'home_team':'team','home_score':'goals_for','away_score':'goals_against'})
team_long_home['points'] = team_long_home['result'].map({'H':3,'D':1,'A':0})
team_long_away = matches_sorted[['date','away_team','home_score','away_score','result']].rename(
    columns={'away_team':'team','away_score':'goals_for','home_score':'goals_against'})
team_long_away['points'] = team_long_away['result'].map({'A':3,'D':1,'H':0})
team_long = pd.concat([team_long_home, team_long_away], ignore_index=True).sort_values(['team','date'])

latest_form = {}
for window in [5, 10]:
    for team, sub in team_long.groupby('team'):
        sub = sub.sort_values('date')
        latest_form.setdefault(team, {})
        latest_form[team][f'form_pts_{window}'] = sub['points'].tail(window).mean()
        latest_form[team][f'goals_for_avg_{window}'] = sub['goals_for'].tail(window).mean()
        latest_form[team][f'goals_against_avg_{window}'] = sub['goals_against'].tail(window).mean()

matches_played_count = team_long.groupby('team').size().to_dict()

h2h_record = {}
for _, row in matches_sorted.iterrows():
    key = tuple(sorted([row['home_team'], row['away_team']]))
    rec = h2h_record.setdefault(key, {'wins': {}, 'draws': 0, 'total': 0})
    rec['total'] += 1
    if row['result'] == 'H': rec['wins'][row['home_team']] = rec['wins'].get(row['home_team'], 0) + 1
    elif row['result'] == 'A': rec['wins'][row['away_team']] = rec['wins'].get(row['away_team'], 0) + 1
    else: rec['draws'] += 1

def get_h2h(home, away):
    key = tuple(sorted([home, away]))
    rec = h2h_record.get(key)
    if not rec or rec['total'] == 0: return 0, np.nan
    return rec['total'], rec['wins'].get(home, 0) / rec['total']

rows = []
for _, fx in fixtures.iterrows():
    h, a = fx['home_team'], fx['away_team']
    r_h, r_a = elo.get(h, INIT_ELO), elo.get(a, INIT_ELO)
    h2h_count, h2h_rate = get_h2h(h, a)
    hf = latest_form.get(h, {})
    af = latest_form.get(a, {})
    home_form_5 = hf.get('form_pts_5', np.nan)
    away_form_5 = af.get('form_pts_5', np.nan)
    home_gf5 = hf.get('goals_for_avg_5', np.nan)
    away_gf5 = af.get('goals_for_avg_5', np.nan)
    row = {
        'date': fx['date'], 'home_team': h, 'away_team': a,
        'elo_diff': r_h - r_a, 'home_elo_pre': r_h, 'away_elo_pre': r_a,
        'neutral': int(fx['neutral']),
        'home_form_pts_5': home_form_5, 'away_form_pts_5': away_form_5,
        'home_form_pts_10': hf.get('form_pts_10', np.nan), 'away_form_pts_10': af.get('form_pts_10', np.nan),
        'home_goals_for_avg_5': home_gf5, 'away_goals_for_avg_5': away_gf5,
        'home_goals_against_avg_5': hf.get('goals_against_avg_5', np.nan),
        'away_goals_against_avg_5': af.get('goals_against_avg_5', np.nan),
        'home_matches_played_before': matches_played_count.get(h, 0),
        'away_matches_played_before': matches_played_count.get(a, 0),
        'h2h_matches_count': h2h_count, 'h2h_home_win_rate': h2h_rate,
    }
    row['abs_elo_diff'] = abs(row['elo_diff'])
    row['elo_close'] = int(row['abs_elo_diff'] < 50)
    row['form_diff'] = (home_form_5 - away_form_5) if pd.notna(home_form_5) and pd.notna(away_form_5) else np.nan
    row['abs_form_diff'] = abs(row['form_diff']) if pd.notna(row['form_diff']) else np.nan
    row['combined_goal_avg'] = ((home_gf5 if pd.notna(home_gf5) else 1) + (away_gf5 if pd.notna(away_gf5) else 1)) / 2
    row['low_scoring_signal'] = int(row['combined_goal_avg'] < 1.2)
    rows.append(row)

pred_df = pd.DataFrame(rows)
X_pred = pred_df[feature_cols]

proba = final_model.predict_proba(X_pred)
preds = final_model.predict(X_pred)
classes = final_model.classes_

out = pred_df[['date','home_team','away_team']].copy()
out['prediction'] = preds
for i, c in enumerate(classes):
    label = {'H':'home_win','D':'draw','A':'away_win'}[c]
    out[f'prob_{label}'] = proba[:, i].round(3)

out.to_csv('fixture_predictions_final.csv', index=False)
print(f"Regenerated predictions for {len(out)} fixtures -> fixture_predictions_final.csv")
print(f"\nDraws predicted: {(preds=='D').sum()} / {len(preds)} fixtures")
print()
print(out.to_string(index=False))
