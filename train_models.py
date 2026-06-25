import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, classification_report

train = pd.read_csv('train_set.csv', parse_dates=['date'])
val = pd.read_csv('validation_2026wc.csv', parse_dates=['date'])

feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

X_train = train[feature_cols].copy()
y_train = train['result']
X_val = val[feature_cols].copy()
y_val = val['result']

X_train['neutral'] = X_train['neutral'].astype(int)
X_val['neutral'] = X_val['neutral'].astype(int)

print("Train class balance:")
print(y_train.value_counts(normalize=True).round(3))
print()
print("Validation (2026 WC) class balance:")
print(y_val.value_counts(normalize=True).round(3))
print()

# =========================================================
# BASELINE: always predict higher-Elo team wins
# =========================================================
def baseline_predict(row):
    if row['elo_diff'] > 0:
        return 'H'
    elif row['elo_diff'] < 0:
        return 'A'
    else:
        return 'D'

baseline_preds = X_val.apply(baseline_predict, axis=1)
baseline_acc = accuracy_score(y_val, baseline_preds)
print(f"BASELINE (higher Elo wins) accuracy: {baseline_acc:.3f}")
print()

# =========================================================
# MODEL 1: Logistic Regression (needs imputation + scaling)
# =========================================================
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

logreg_preds = logreg.predict(X_val_scaled)
logreg_proba = logreg.predict_proba(X_val_scaled)
logreg_acc = accuracy_score(y_val, logreg_preds)
logreg_ll = log_loss(y_val, logreg_proba, labels=logreg.classes_)

print(f"LOGISTIC REGRESSION — accuracy: {logreg_acc:.3f}, log-loss: {logreg_ll:.3f}")
print()

# =========================================================
# MODEL 2: HistGradientBoostingClassifier (sklearn's XGBoost equivalent)
# Native NaN support -- no imputation needed
# =========================================================
hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
hgb.fit(X_train, y_train)

hgb_preds = hgb.predict(X_val)
hgb_proba = hgb.predict_proba(X_val)
hgb_acc = accuracy_score(y_val, hgb_preds)
hgb_ll = log_loss(y_val, hgb_proba, labels=hgb.classes_)

print(f"GRADIENT BOOSTING (HGB) — accuracy: {hgb_acc:.3f}, log-loss: {hgb_ll:.3f}")
print()

# =========================================================
# Side by side comparison
# =========================================================
print("="*60)
print("SUMMARY — 2026 World Cup validation set (20 matches)")
print("="*60)
print(f"{'Model':<30}{'Accuracy':<12}{'Log-loss':<10}")
print(f"{'Baseline (Elo only)':<30}{baseline_acc:<12.3f}{'-':<10}")
print(f"{'Logistic Regression':<30}{logreg_acc:<12.3f}{logreg_ll:<10.3f}")
print(f"{'Gradient Boosting':<30}{hgb_acc:<12.3f}{hgb_ll:<10.3f}")
print()

print("Detailed predictions vs actual (Gradient Boosting):")
comparison = val[['date','home_team','away_team','result']].copy()
comparison['hgb_pred'] = hgb_preds
comparison['logreg_pred'] = logreg_preds
comparison['baseline_pred'] = baseline_preds.values
comparison['hgb_correct'] = comparison['hgb_pred'] == comparison['result']
print(comparison.to_string(index=False))

# Save models output for later use
import joblib
joblib.dump(hgb, 'model_hgb.pkl')
joblib.dump(logreg, 'model_logreg.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
comparison.to_csv('validation_predictions.csv', index=False)
