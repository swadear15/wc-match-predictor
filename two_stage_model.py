import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, precision_score, recall_score
import joblib

train = pd.read_csv('train_set.csv', parse_dates=['date'])
val = pd.read_csv('validation_2026wc.csv', parse_dates=['date'])

base_feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

for df in [train, val]:
    df['abs_elo_diff'] = df['elo_diff'].abs()
    df['elo_close'] = (df['abs_elo_diff'] < 50).astype(int)
    df['form_diff'] = df['home_form_pts_5'] - df['away_form_pts_5']
    df['abs_form_diff'] = df['form_diff'].abs()
    df['combined_goal_avg'] = (df['home_goals_for_avg_5'].fillna(1) + df['away_goals_for_avg_5'].fillna(1)) / 2
    df['low_scoring_signal'] = (df['combined_goal_avg'] < 1.2).astype(int)

feature_cols = base_feature_cols + ['abs_elo_diff', 'elo_close', 'form_diff', 'abs_form_diff',
                                      'combined_goal_avg', 'low_scoring_signal']

X_train = train[feature_cols].copy()
X_train['neutral'] = X_train['neutral'].astype(int)
y_train_3class = train['result']
y_train_draw = (train['result'] == 'D').astype(int)  # binary: is draw?

X_val = val[feature_cols].copy()
X_val['neutral'] = X_val['neutral'].astype(int)
y_val_3class = val['result']
y_val_draw = (val['result'] == 'D').astype(int)

print("="*70)
print("STAGE 1: Draw vs No-Draw binary classifier")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

draw_class_weight = {0: 1.0, 1: (y_train_draw==0).sum() / (y_train_draw==1).sum()}
print(f"Draw class weight (to balance ~23% positive rate): {draw_class_weight}")

fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []
for tr_idx, te_idx in cv.split(X_train, y_train_draw):
    X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
    y_tr, y_te = y_train_draw.iloc[tr_idx], y_train_draw.iloc[te_idx]
    sw = y_tr.map(draw_class_weight).values
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    m.fit(X_tr, y_tr, sample_weight=sw)
    preds = m.predict(X_te)
    fold_accs.append(accuracy_score(y_te, preds))
    fold_f1s.append(f1_score(y_te, preds, zero_division=0))
    fold_precs.append(precision_score(y_te, preds, zero_division=0))
    fold_recs.append(recall_score(y_te, preds, zero_division=0))

print(f"\nCV accuracy: {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs):.3f})")
print(f"CV F1 (draw class): {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")
print(f"CV precision (draw class): {np.mean(fold_precs):.3f}")
print(f"CV recall (draw class): {np.mean(fold_recs):.3f}")
print("(Recall = of all actual draws, what fraction did we catch)")
print("(Precision = of matches we called 'draw', what fraction were actually draws)")

# Fit final draw classifier on full train
sw_full = y_train_draw.map(draw_class_weight).values
draw_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
draw_clf.fit(X_train, y_train_draw, sample_weight=sw_full)

print("\n" + "="*70)
print("STAGE 2: Home-win vs Away-win classifier (trained ONLY on non-draw matches)")
print("="*70)

non_draw_mask = y_train_3class != 'D'
X_train_nd = X_train[non_draw_mask]
y_train_nd = (y_train_3class[non_draw_mask] == 'H').astype(int)  # 1=Home win, 0=Away win

fold_accs_wl = []
for tr_idx, te_idx in cv.split(X_train_nd, y_train_nd):
    X_tr, X_te = X_train_nd.iloc[tr_idx], X_train_nd.iloc[te_idx]
    y_tr, y_te = y_train_nd.iloc[tr_idx], y_train_nd.iloc[te_idx]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)
    fold_accs_wl.append(accuracy_score(y_te, preds))

print(f"CV accuracy (home vs away, non-draw matches only): {np.mean(fold_accs_wl):.3f} (+/- {np.std(fold_accs_wl):.3f})")

wl_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
wl_clf.fit(X_train_nd, y_train_nd)

print("\n" + "="*70)
print("COMBINED TWO-STAGE PIPELINE: evaluated on 20-game validation set")
print("="*70)

draw_proba_val = draw_clf.predict_proba(X_val)[:, 1]  # P(draw)
draw_pred_val = draw_clf.predict(X_val)

wl_proba_val = wl_clf.predict_proba(X_val)[:, 1]  # P(home win | not draw)

final_preds = []
final_probs = []  # (P_A, P_D, P_H) reconstructed
for i in range(len(X_val)):
    p_draw = draw_proba_val[i]
    if draw_pred_val[i] == 1:
        final_preds.append('D')
    else:
        p_home_given_not_draw = wl_proba_val[i]
        final_preds.append('H' if p_home_given_not_draw >= 0.5 else 'A')
    # reconstruct full 3-class probability for log-loss comparison
    p_h = (1 - p_draw) * wl_proba_val[i]
    p_a = (1 - p_draw) * (1 - wl_proba_val[i])
    final_probs.append([p_a, p_draw, p_h])  # order A, D, H to match alphabetical

final_preds = np.array(final_preds)
final_probs = np.array(final_probs)

acc = accuracy_score(y_val_3class, final_preds)
f1 = f1_score(y_val_3class, final_preds, average='macro', zero_division=0)
ll = log_loss(y_val_3class, final_probs, labels=['A','D','H'])

print(f"\nAccuracy: {acc:.3f} | Macro-F1: {f1:.3f} | Log-loss: {ll:.3f}")

classes = ['A','D','H']
class_names = {'H':'Home win','D':'Draw','A':'Away win'}
cm = confusion_matrix(y_val_3class, final_preds, labels=classes)
print(f"\nConfusion matrix:")
header = "".join([f"{class_names[c]:<12}" for c in classes])
print(f"{'':<14}{header}")
for i, c in enumerate(classes):
    row = "".join([f"{cm[i][j]:<12}" for j in range(len(classes))])
    print(f"{class_names[c]:<14}{row}")

n_draws_predicted = (final_preds == 'D').sum()
n_draws_correct = ((final_preds == 'D') & (y_val_3class == 'D')).sum()
print(f"\nDraws predicted: {n_draws_predicted}/20 | Correct draws: {n_draws_correct}/8 actual draws")

comparison = val[['date','home_team','away_team','result']].copy()
comparison['two_stage_pred'] = final_preds
comparison['prob_draw'] = draw_proba_val.round(3)
comparison['prob_home_given_not_draw'] = wl_proba_val.round(3)
print("\n" + comparison.to_string(index=False))

joblib.dump(draw_clf, 'model_draw_classifier.pkl')
joblib.dump(wl_clf, 'model_winloss_classifier.pkl')
comparison.to_csv('validation_predictions_two_stage.csv', index=False)
print("\nSaved model_draw_classifier.pkl, model_winloss_classifier.pkl, validation_predictions_two_stage.csv")
