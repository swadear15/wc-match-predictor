import pandas as pd
import numpy as np

# Load all sources
results = pd.read_csv('/mnt/user-data/uploads/results.csv', parse_dates=['date'])
shootouts = pd.read_csv('/mnt/user-data/uploads/shootouts.csv', parse_dates=['date'])
goalscorers = pd.read_csv('/mnt/user-data/uploads/goalscorers.csv', parse_dates=['date'])
former_names = pd.read_csv('/mnt/user-data/uploads/former_names.csv')

# --- 1. Split played vs upcoming (future fixtures have no score yet) ---
played = results[results.home_score.notna()].copy()
upcoming = results[results.home_score.isna()].copy()

played['home_score'] = played['home_score'].astype(int)
played['away_score'] = played['away_score'].astype(int)

# --- 2. Resolve draws using shootout winner (knockout matches) ---
# Merge shootout results onto played matches sharing date+teams
shootouts_key = shootouts.rename(columns={'winner': 'shootout_winner'})
played = played.merge(
    shootouts_key[['date', 'home_team', 'away_team', 'shootout_winner']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)

# --- 3. Create match outcome target ---
# result: 'H' home win, 'A' away win, 'D' draw (regulation, before any shootout)
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

played['result'] = played.apply(get_result, axis=1)

# Knockout-adjusted winner: if drawn but decided by shootout, record who actually advanced
def get_match_winner(row):
    if row['result'] != 'D':
        return row['home_team'] if row['result'] == 'H' else row['away_team']
    if pd.notna(row['shootout_winner']):
        return row['shootout_winner']
    return None  # genuine draw, no winner

played['match_winner'] = played.apply(get_match_winner, axis=1)
played['decided_by_shootout'] = played['shootout_winner'].notna()

# --- 4. Aggregate goalscorer data into per-match summary stats ---
# (not goal-by-goal, just useful match-level signals)
gs_summary = goalscorers.groupby(['date', 'home_team', 'away_team']).agg(
    total_goals_logged=('scorer', 'count'),
    penalty_goals=('penalty', 'sum'),
    own_goals=('own_goal', 'sum'),
    last_goal_minute=('minute', 'max')
).reset_index()

played = played.merge(gs_summary, on=['date', 'home_team', 'away_team'], how='left')

# --- 5. Sort chronologically (critical for Elo / rolling features later) ---
played = played.sort_values('date').reset_index(drop=True)

# --- 6. Flag World Cup matches specifically ---
played['is_world_cup'] = played['tournament'] == 'FIFA World Cup'
upcoming['is_world_cup'] = upcoming['tournament'] == 'FIFA World Cup'

# Save outputs
played.to_csv('/home/claude/wc_project/matches_played.csv', index=False)
upcoming.to_csv('/home/claude/wc_project/fixtures_upcoming.csv', index=False)

print("=== matches_played.csv ===")
print(f"Shape: {played.shape}")
print(f"Date range: {played.date.min().date()} to {played.date.max().date()}")
print(f"Columns: {played.columns.tolist()}")
print()
print("=== fixtures_upcoming.csv (2026 WC predictions targets) ===")
print(f"Shape: {upcoming.shape}")
print(upcoming[['date','home_team','away_team','tournament']].head(10))
print()
print("Sample of played data:")
print(played[['date','home_team','away_team','home_score','away_score','result','match_winner','decided_by_shootout','is_world_cup']].tail(10))
