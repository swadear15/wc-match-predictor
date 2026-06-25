import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
import joblib

train = pd.read_csv('train_set.csv', parse_dates=['date'])
val = pd.read_csv('validation_2026wc.csv', parse_dates=['date'])

base_feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

# =========================================================
# NEW FEATURE: closeness signals (predict draws, not just winner)
# =========================================================
for df in [train, val]:
    df['abs_elo_diff'] = df['elo_diff'].abs()
    df['elo_close'] = (df['abs_elo_diff'] < 50).astype(int)  # very evenly matched
    df['form_diff'] = df['home_form_pts_5'] - df['away_form_pts_5']
    df['abs_form_diff'] = df['form_diff'].abs()
    df['combined_goal_avg'] = (df['home_goals_for_avg_5'].fillna(1) + df['away_goals_for_avg_5'].fillna(1)) / 2
    # low-scoring matchups draw more often
    df['low_scoring_signal'] = (df['combined_goal_avg'] < 1.2).astype(int)

feature_cols = base_feature_cols + ['abs_elo_diff', 'elo_close', 'form_diff', 'abs_form_diff',
                                      'combined_goal_avg', 'low_scoring_signal']

X_train = train[feature_cols].copy()
y_train = train['result']
X_train['neutral'] = X_train['neutral'].astype(int)

X_val = val[feature_cols].copy()
y_val = val['result']
X_val['neutral'] = X_val['neutral'].astype(int)

classes = sorted(y_train.unique())

print("="*70)
print("STEP 1: Cross-validated comparison on TRAINING set (5-fold, stratified)")
print("(more reliable than 20-game validation accuracy alone)")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline HGB (no class weight) for comparison
hgb_plain = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
cv_acc_plain = cross_val_score(hgb_plain, X_train, y_train, cv=cv, scoring='accuracy')
cv_f1_plain = cross_val_score(hgb_plain, X_train, y_train, cv=cv, scoring='f1_macro')
print(f"\nHGB (no weighting, new features) -- CV accuracy: {cv_acc_plain.mean():.3f} (+/- {cv_acc_plain.std():.3f})")
print(f"HGB (no weighting, new features) -- CV macro-F1:  {cv_f1_plain.mean():.3f} (+/- {cv_f1_plain.std():.3f})")

# =========================================================
# STEP 2: Class weighting to combat draw under-prediction
# =========================================================
# sklearn HGB doesn't take class_weight directly pre-1.3 in all configs, but it does via sample_weight
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {c: total / (len(class_counts) * class_counts[c]) for c in class_counts.index}
print(f"\nComputed balanced class weights: {class_weights}")
sample_weight_train = y_train.map(class_weights).values

hgb_weighted = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)

# cross_val_score doesn't easily pass sample_weight per-fold without a wrapper; do manual CV
fold_accs, fold_f1s = [], []
for train_idx, test_idx in cv.split(X_train, y_train):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
    sw_tr = y_tr.map(class_weights).values
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    m.fit(X_tr, y_tr, sample_weight=sw_tr)
    preds = m.predict(X_te)
    fold_accs.append(accuracy_score(y_te, preds))
    fold_f1s.append(f1_score(y_te, preds, average='macro', zero_division=0))

print(f"\nHGB (class-weighted, new features) -- CV accuracy: {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs):.3f})")
print(f"HGB (class-weighted, new features) -- CV macro-F1:  {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")

# =========================================================
# STEP 3: Fit final weighted model on full train, evaluate on validation
# =========================================================
hgb_final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
hgb_final.fit(X_train, y_train, sample_weight=sample_weight_train)

val_preds = hgb_final.predict(X_val)
val_proba = hgb_final.predict_proba(X_val)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
val_ll = log_loss(y_val, val_proba, labels=hgb_final.classes_)

print("\n" + "="*70)
print("STEP 3: Final weighted+featured model on 20-game validation set")
print("="*70)
print(f"Accuracy: {val_acc:.3f} | Macro-F1: {val_f1:.3f} | Log-loss: {val_ll:.3f}")

cm = confusion_matrix(y_val, val_preds, labels=classes)
class_names = {'H':'Home win','D':'Draw','A':'Away win'}
print(f"\nConfusion matrix:")
header = "".join([f"{class_names[c]:<12}" for c in classes])
print(f"{'':<14}{header}")
for i, c in enumerate(classes):
    row = "".join([f"{cm[i][j]:<12}" for j in range(len(classes))])
    print(f"{class_names[c]:<14}{row}")

n_draws_predicted = (val_preds == 'D').sum()
print(f"\nDraws predicted: {n_draws_predicted} (out of 20 matches, 8 actual draws)")

comparison = val[['date','home_team','away_team','result']].copy()
comparison['weighted_hgb_pred'] = val_preds
for i, c in enumerate(hgb_final.classes_):
    comparison[f'prob_{c}'] = val_proba[:, i].round(3)
print("\n" + comparison.to_string(index=False))

joblib.dump(hgb_final, 'model_hgb_weighted.pkl')
comparison.to_csv('validation_predictions_weighted.csv', index=False)

print("\nSaved model_hgb_weighted.pkl and validation_predictions_weighted.csv")
