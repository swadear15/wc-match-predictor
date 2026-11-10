"""
tournament.py -- World Cup structure: groups, tie-breakers, 2026 bracket, and the
historical group-stage -> knockout dataset used to fit the playoff model.
"""
import random
from collections import defaultdict
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# 2026 format: 12 groups of 4; top 2 + 8 best third-placed teams -> 32-team knockout.
# Each group contains exactly one top-pot team / host, which we use to label the groups.
GROUP_ANCHORS_2026 = {
    'A': 'Mexico', 'B': 'Canada', 'C': 'Brazil', 'D': 'United States', 'E': 'Germany',
    'F': 'Netherlands', 'G': 'Belgium', 'H': 'Spain', 'I': 'France', 'J': 'Argentina',
    'K': 'Portugal', 'L': 'England',
}

# Round of 32 (FIFA match numbers 73-88): slot -> '1X' group winner, '2X' runner-up,
# '3:ABC..' = a third-placed team from one of the listed groups.
R32 = {
    73: ('2A', '2B'), 74: ('1E', '3:ABCDF'), 75: ('1F', '2C'), 76: ('1C', '2F'),
    77: ('1I', '3:CDFGH'), 78: ('2E', '2I'), 79: ('1A', '3:CEFHI'), 80: ('1L', '3:EHIJK'),
    81: ('1D', '3:BEFIJ'), 82: ('1G', '3:AEHIJ'), 83: ('2K', '2L'), 84: ('1H', '2J'),
    85: ('1B', '3:EFGIJ'), 86: ('1J', '2H'), 87: ('1K', '3:DEIJL'), 88: ('2D', '2G'),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
THIRD_SLOT_MATCHES = [74, 77, 79, 80, 81, 82, 85, 87]


_MATCHING_CACHE = {}


def third_place_matchings(groups_qualified):
    """All ways to assign the 8 qualifying third-placed groups to the 8 R32 slots that take a
    third-placed team (each slot only accepts the groups listed in R32). FIFA publishes ONE
    fixed assignment per combination of qualifying groups (a 495-row table). That table is not
    in the dataset, so the simulator draws uniformly among the valid assignments, which keeps
    the group-winner / runner-up pairings exact and is unbiased for the third-place ones.
    Returns a list of {match_number: group_letter}."""
    key = frozenset(groups_qualified)
    if key in _MATCHING_CACHE:
        return _MATCHING_CACHE[key]
    groups = sorted(key)
    slots = THIRD_SLOT_MATCHES
    elig = {m: set(R32[m][1].split(':')[1]) for m in slots}
    out = []

    def rec(i, used, cur):
        if i == len(slots):
            out.append(dict(cur))
            return
        m = slots[i]
        for g in groups:
            if g in elig[m] and g not in used:
                cur[m] = g
                rec(i + 1, used | {g}, cur)
                del cur[m]

    rec(0, frozenset(), {})
    _MATCHING_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------------
def _table(subset, games):
    t = {x: [0, 0, 0] for x in subset}           # points, goal diff, goals for
    for a, b, ga, gb in games:
        if a in t and b in t:
            t[a][1] += ga - gb
            t[b][1] += gb - ga
            t[a][2] += ga
            t[b][2] += gb
            if ga > gb:
                t[a][0] += 3
            elif ga < gb:
                t[b][0] += 3
            else:
                t[a][0] += 1
                t[b][0] += 1
    return t


def rank_group(teams, games, rng=random):
    """Order a group best -> worst. FIFA 2026 criteria: points; then head-to-head points,
    goal difference, goals among the tied teams; then overall goal difference and goals;
    then (fair play / ranking, unavailable here) a random draw.
    games: iterable of (team_a, team_b, goals_a, goals_b)."""
    games = list(games)
    overall = _table(teams, games)
    order = sorted(teams, key=lambda x: -overall[x][0])
    out, i = [], 0
    while i < len(order):
        j = i
        while j < len(order) and overall[order[j]][0] == overall[order[i]][0]:
            j += 1
        tied = order[i:j]
        if len(tied) > 1:
            h = _table(tied, games)
            tied = sorted(tied, key=lambda x: (-h[x][0], -h[x][1], -h[x][2],
                                               -overall[x][1], -overall[x][2], rng.random()))
        out += tied
        i = j
    return out


def derive_groups(group_matches):
    """Connected components of 'played each other' -> list of sets of teams."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in group_matches.itertuples():
        parent[find(r.home_team)] = find(r.away_team)
    comps = defaultdict(set)
    for t in list(parent):
        comps[find(t)].add(t)
    return list(comps.values())


def label_groups_2026(group_sets):
    """Map letter -> ordered list of teams, using the anchor team of each group."""
    out = {}
    for letter, anchor in GROUP_ANCHORS_2026.items():
        for g in group_sets:
            if anchor in g:
                out[letter] = sorted(g)
    assert len(out) == 12 and sum(len(v) for v in out.values()) == 48, 'group labelling failed'
    return out


# ----------------------------------------------------------------------------------
# Historical editions: group stage vs knockout split (World Cups since 1986)
def edition(d, year):
    """Return (group_stage_matches, knockout_matches) of a World Cup edition.
    From 1986 the last 16 matches (32 from 2026) are the knockout rounds, incl. 3rd place."""
    w = d[(d.tournament == 'FIFA World Cup') & (d.date.dt.year == year)]
    w = w.sort_values('date', kind='stable')
    n_ko = 32 if year >= 2026 else 16
    return w.iloc[:-n_ko].copy(), w.iloc[-n_ko:].copy()


def shootout_map(path='shootouts.csv'):
    s = pd.read_csv(path, parse_dates=['date'])
    return {(r.date, r.home_team, r.away_team): r.winner for r in s.itertuples()}


def advancing_team(row, smap):
    if row.home_score > row.away_score:
        return row.home_team
    if row.home_score < row.away_score:
        return row.away_team
    return smap.get((row.date, row.home_team, row.away_team))


def group_stage_stats(gs):
    """Per-team group-stage summary from a frame of group matches that has we_h (pre-match
    expected score of the home side) -- 'overperf' is actual minus Elo-expected points share."""
    st = defaultdict(lambda: dict(gp=0, pts=0, gd=0, gf=0, ga=0, overperf=0.0))
    for r in gs.itertuples():
        w_h = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
        for team, gf, ga, w, we in ((r.home_team, r.home_score, r.away_score, w_h, r.we_h),
                                    (r.away_team, r.away_score, r.home_score, 1 - w_h, 1 - r.we_h)):
            s = st[team]
            s['gp'] += 1
            s['gf'] += gf
            s['ga'] += ga
            s['gd'] += gf - ga
            s['pts'] += 3 if gf > ga else (1 if gf == ga else 0)
            s['overperf'] += w - we
    groups = derive_groups(gs)
    for g in groups:
        games = [(r.home_team, r.away_team, r.home_score, r.away_score)
                 for r in gs.itertuples() if r.home_team in g]
        order = rank_group(sorted(g), games, rng=random.Random(0))
        for pos, t in enumerate(order, 1):
            st[t]['rank'] = pos
    return st


def knockout_dataset(d, years, smap):
    """One row per knockout match (3rd-place game excluded) with Elo and group-stage features.
    Features are differences team1 - team2, team1 = the file's 'home_team'. y = team1 advances."""
    rows = []
    for yr in years:
        gs, ko = edition(d, yr)
        if len(gs) == 0:
            continue
        st = group_stage_stats(gs)
        first_elo = {}
        for r in gs.itertuples():
            first_elo.setdefault(r.home_team, r.elo_h)
            first_elo.setdefault(r.away_team, r.elo_a)
        ko = ko.reset_index(drop=True)
        for i, r in enumerate(ko.itertuples()):
            if i == len(ko) - 2:          # third-place match: different incentives, skip
                continue
            w = advancing_team(r, smap)
            if w is None:
                continue
            a, b = r.home_team, r.away_team
            sa, sb = st.get(a), st.get(b)
            if sa is None or sb is None:
                continue
            pa, pb = max(sa['gp'], 1), max(sb['gp'], 1)
            rows.append(dict(
                year=yr, date=r.date, team1=a, team2=b, y=int(w == a),
                d_elo=r.elo_diff_adj / 100.0,
                d_elo_pre=(first_elo[a] - first_elo[b]) / 100.0,
                d_overperf=sa['overperf'] / pa - sb['overperf'] / pb,
                d_gd_pg=sa['gd'] / pa - sb['gd'] / pb,
                d_ppg=sa['pts'] / pa - sb['pts'] / pb,
                d_gf_pg=sa['gf'] / pa - sb['gf'] / pb,
                d_won_group=int(sa['rank'] == 1) - int(sb['rank'] == 1),
            ))
    return pd.DataFrame(rows)
