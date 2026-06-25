import pandas as pd
import numpy as np

played = pd.read_csv('matches_played.csv', parse_dates=['date'])
played = played.sort_values('date').reset_index(drop=True)

# --- Split off the 20 already-played 2026 WC matches as validation ---
wc2026_played = played[(played.is_world_cup) & (played.date >= '2026-01-01')].copy()
train_pool = played[~played.index.isin(wc2026_played.index)].copy()

print(f"Training pool: {len(train_pool)} matches")
print(f"2026 WC validation: {len(wc2026_played)} matches")
print()

# We still need ALL matches in chronological order to compute Elo/form correctly
# (a team's Elo going into the 2026 WC depends on everything before it, including
# other 2026 qualifiers/friendlies not in the WC group itself).
# So: compute features over the FULL played set, then split by index afterward.

# =========================================================
# 1. ELO RATING SYSTEM
# =========================================================
K = 20  # sensitivity factor (lower = more stable, less reactive)
HOME_ADV = 0  # set 0 since 'neutral' already flags neutral-site games; we'll handle separately

elo = {}  # team -> current rating
INIT_ELO = 1500

def get_elo(team):
    return elo.get(team, INIT_ELO)

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

# Goal-difference multiplier (standard World Football Elo approach)
def goal_diff_multiplier(gd):
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8

home_elo_pre = []
away_elo_pre = []

for idx, row in played.iterrows():
    h, a = row['home_team'], row['away_team']
    r_h, r_a = get_elo(h), get_elo(a)
    home_elo_pre.append(r_h)
    away_elo_pre.append(r_a)

    # actual score for Elo update purposes (shootout draws count as 0.5/0.5 — going to extra time
    # means the match was genuinely even in regulation)
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

    new_r_h = r_h + K * mult * (s_h - exp_h)
    new_r_a = r_a + K * mult * (s_a - exp_a)

    elo[h] = new_r_h
    elo[a] = new_r_a

played['home_elo_pre'] = home_elo_pre
played['away_elo_pre'] = away_elo_pre
played['elo_diff'] = played['home_elo_pre'] - played['away_elo_pre']

print("Elo computed. Sample final ratings (top 10):")
final_elo = pd.Series(elo).sort_values(ascending=False)
print(final_elo.head(10))
print()

# =========================================================
# 2. ROLLING FORM FEATURES (last N matches per team, no leakage)
# =========================================================
# Build a long-format team-match table: one row per team per match
home_rows = played[['date','home_team','away_team','home_score','away_score','result']].copy()
home_rows['team'] = home_rows['home_team']
home_rows['opponent'] = home_rows['away_team']
home_rows['goals_for'] = home_rows['home_score']
home_rows['goals_against'] = home_rows['away_score']
home_rows['points'] = home_rows['result'].map({'H': 3, 'D': 1, 'A': 0})
home_rows['match_idx'] = home_rows.index

away_rows = played[['date','home_team','away_team','home_score','away_score','result']].copy()
away_rows['team'] = away_rows['away_team']
away_rows['opponent'] = away_rows['home_team']
away_rows['goals_for'] = away_rows['away_score']
away_rows['goals_against'] = away_rows['home_score']
away_rows['points'] = away_rows['result'].map({'A': 3, 'D': 1, 'H': 0})
away_rows['match_idx'] = away_rows.index

team_long = pd.concat([home_rows, away_rows], ignore_index=True)
team_long = team_long.sort_values(['team', 'date']).reset_index(drop=True)

# Rolling windows computed using only PRIOR matches (shift(1) before rolling)
for window in [5, 10]:
    team_long[f'form_pts_{window}'] = (
        team_long.groupby('team')['points']
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(drop=True)
    )
    team_long[f'goals_for_avg_{window}'] = (
        team_long.groupby('team')['goals_for']
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(drop=True)
    )
    team_long[f'goals_against_avg_{window}'] = (
        team_long.groupby('team')['goals_against']
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(drop=True)
    )

# Matches played so far (experience / data-richness signal)
team_long['matches_played_before'] = team_long.groupby('team').cumcount()

form_cols = [c for c in team_long.columns if c.startswith('form_') or c.startswith('goals_for_avg') or c.startswith('goals_against_avg')] + ['matches_played_before']

# Pivot back: merge home-team form and away-team form onto played
home_form = team_long[team_long['team'] == team_long['home_team']][['match_idx'] + form_cols].copy()
home_form.columns = ['match_idx'] + ['home_' + c for c in form_cols]

away_form = team_long[team_long['team'] == team_long['away_team']][['match_idx'] + form_cols].copy()
away_form.columns = ['match_idx'] + ['away_' + c for c in form_cols]

played = played.merge(home_form, left_index=True, right_on='match_idx', how='left').drop(columns='match_idx')
played = played.merge(away_form, left_index=True, right_on='match_idx', how='left').drop(columns='match_idx')

print("Rolling form features added:", [c for c in played.columns if 'form_' in c or 'goals_for_avg' in c or 'goals_against_avg' in c])
print()

# =========================================================
# 3. HEAD-TO-HEAD RECORD (prior meetings only)
# =========================================================
played['h2h_key'] = played.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
played = played.sort_values('date').reset_index(drop=True)

h2h_home_wins = {}
h2h_away_wins = {}
h2h_draws = {}
h2h_home_win_rate = []
h2h_matches_count = []

for idx, row in played.iterrows():
    key = row['h2h_key']
    h, a = row['home_team'], row['away_team']
    total = h2h_home_wins.get(key, 0) + h2h_away_wins.get(key, 0) + h2h_draws.get(key, 0)
    h2h_matches_count.append(total)

    if total == 0:
        h2h_home_win_rate.append(np.nan)
    else:
        # win rate of CURRENT home team in this fixture, from prior meetings
        team_a_wins = h2h_home_wins.get(key, 0) if key[0] == h else h2h_away_wins.get(key, 0)
        h2h_home_win_rate.append(team_a_wins / total)

    # update after recording pre-match state
    if row['result'] == 'H':
        if key[0] == h:
            h2h_home_wins[key] = h2h_home_wins.get(key, 0) + 1
        else:
            h2h_away_wins[key] = h2h_away_wins.get(key, 0) + 1
    elif row['result'] == 'A':
        if key[0] == a:
            h2h_home_wins[key] = h2h_home_wins.get(key, 0) + 1
        else:
            h2h_away_wins[key] = h2h_away_wins.get(key, 0) + 1
    else:
        h2h_draws[key] = h2h_draws.get(key, 0) + 1

played['h2h_matches_count'] = h2h_matches_count
played['h2h_home_win_rate'] = h2h_home_win_rate

print("H2H features added. Sample:")
print(played[['date','home_team','away_team','h2h_matches_count','h2h_home_win_rate']].tail(5))
print()

# =========================================================
# Save full featured dataset + re-split train/validation
# =========================================================
played.to_csv('matches_featured.csv', index=False)

train_final = played[~((played.is_world_cup) & (played.date >= '2026-01-01'))]
val_final = played[(played.is_world_cup) & (played.date >= '2026-01-01')]

train_final.to_csv('train_set.csv', index=False)
val_final.to_csv('validation_2026wc.csv', index=False)

print(f"Final train set: {len(train_final)} rows -> train_set.csv")
print(f"Final validation set (2026 WC played): {len(val_final)} rows -> validation_2026wc.csv")
print()
print("Feature columns available for modelling:")
feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']
print(feature_cols)
print()
print("Missing values in feature set (train):")
print(train_final[feature_cols].isna().sum())
