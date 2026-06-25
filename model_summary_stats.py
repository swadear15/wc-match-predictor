import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# =========================================================
# Load the FINAL shipped model (class-weighted HGB, 3-class)
# =========================================================
final_model = joblib.load('model_final.pkl')

train = pd.read_csv('train_set.csv', parse_dates=['date'])
val = pd.read_csv('validation_2026wc.csv', parse_dates=['date'])

base_feature_cols = ['elo_diff', 'home_elo_pre', 'away_elo_pre', 'neutral',
                 'home_form_pts_5','away_form_pts_5','home_form_pts_10','away_form_pts_10',
                 'home_goals_for_avg_5','away_goals_for_avg_5',
                 'home_goals_against_avg_5','away_goals_against_avg_5',
                 'home_matches_played_before','away_matches_played_before',
                 'h2h_matches_count','h2h_home_win_rate']

# Same draw-detection features added in finalize_model.py -- must match exactly
# or the model will see a different feature set than it was trained on.
def add_draw_features(df):
    df['abs_elo_diff'] = df['elo_diff'].abs()
    df['elo_close'] = (df['abs_elo_diff'] < 50).astype(int)
    df['form_diff'] = df['home_form_pts_5'] - df['away_form_pts_5']
    df['abs_form_diff'] = df['form_diff'].abs()
    df['combined_goal_avg'] = (df['home_goals_for_avg_5'].fillna(1) + df['away_goals_for_avg_5'].fillna(1)) / 2
    df['low_scoring_signal'] = (df['combined_goal_avg'] < 1.2).astype(int)
    return df

train = add_draw_features(train)
val = add_draw_features(val)

feature_cols = base_feature_cols + ['abs_elo_diff', 'elo_close', 'form_diff', 'abs_form_diff',
                                      'combined_goal_avg', 'low_scoring_signal']

X_train = train[feature_cols].copy()
X_train['neutral'] = X_train['neutral'].astype(int)
y_train = train['result']

X_val = val[feature_cols].copy()
X_val['neutral'] = X_val['neutral'].astype(int)
y_val = val['result']

classes = sorted(y_train.unique())  # ['A','D','H']
class_names = {'H': 'Home win', 'D': 'Draw', 'A': 'Away win'}

# =========================================================
# Full evaluation report
# =========================================================
preds_tr = final_model.predict(X_train)
proba_tr = final_model.predict_proba(X_train)
preds_va = final_model.predict(X_val)
proba_va = final_model.predict_proba(X_val)

print("="*70)
print("  FINAL MODEL — class-weighted HistGradientBoosting (3-class)")
print("="*70)

acc_tr = accuracy_score(y_train, preds_tr)
acc_va = accuracy_score(y_val, preds_va)
ll_tr = log_loss(y_train, proba_tr, labels=final_model.classes_)
ll_va = log_loss(y_val, proba_va, labels=final_model.classes_)

print(f"\n{'Metric':<28}{'Train (49,405)':<18}{'Validation (20)':<18}")
print(f"{'-'*64}")
print(f"{'Accuracy':<28}{acc_tr:<18.3f}{acc_va:<18.3f}")
print(f"{'Log-loss':<28}{ll_tr:<18.3f}{ll_va:<18.3f}")

gap = acc_tr - acc_va
flag = "  <-- overfit warning" if gap > 0.15 else ""
print(f"{'Train-val accuracy gap':<28}{gap:<18.3f}{'':<18}{flag}")

# --- Per-class precision/recall/f1 on validation ---
print(f"\nPer-class performance (validation set):")
print(f"{'Class':<14}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}")
for c in classes:
    prec = precision_score(y_val, preds_va, labels=[c], average='micro', zero_division=0)
    rec = recall_score(y_val, preds_va, labels=[c], average='micro', zero_division=0)
    f1 = f1_score(y_val, preds_va, labels=[c], average='micro', zero_division=0)
    support = (y_val == c).sum()
    print(f"{class_names[c]:<14}{prec:<12.3f}{rec:<12.3f}{f1:<12.3f}{support:<10}")

macro_f1 = f1_score(y_val, preds_va, average='macro', zero_division=0)
weighted_f1 = f1_score(y_val, preds_va, average='weighted', zero_division=0)
print(f"{'Macro avg F1':<14}{'':<12}{'':<12}{macro_f1:<12.3f}")
print(f"{'Weighted F1':<14}{'':<12}{'':<12}{weighted_f1:<12.3f}")

# --- Confusion matrix ---
cm = confusion_matrix(y_val, preds_va, labels=classes)
print(f"\nConfusion matrix (validation, rows=actual, cols=predicted):")
header = "".join([f"{class_names[c]:<12}" for c in classes])
print(f"{'':<14}{header}")
for i, c in enumerate(classes):
    row = "".join([f"{cm[i][j]:<12}" for j in range(len(classes))])
    print(f"{class_names[c]:<14}{row}")

# --- One-vs-rest ROC-AUC per class ---
y_val_bin = label_binarize(y_val, classes=final_model.classes_)
print(f"\nOne-vs-rest ROC-AUC (validation):")
for i, c in enumerate(final_model.classes_):
    try:
        auc = roc_auc_score(y_val_bin[:, i], proba_va[:, i])
        print(f"  {class_names[c]:<14}{auc:.3f}")
    except ValueError:
        print(f"  {class_names[c]:<14}n/a (insufficient class examples)")

# --- Baseline for comparison ---
def baseline_predict(row):
    if row['elo_diff'] > 0: return 'H'
    elif row['elo_diff'] < 0: return 'A'
    else: return 'D'

baseline_preds_val = X_val.apply(baseline_predict, axis=1)
baseline_acc = accuracy_score(y_val, baseline_preds_val)
baseline_macro_f1 = f1_score(y_val, baseline_preds_val, average='macro', zero_division=0)

print("\n" + "="*70)
print("  SUMMARY TABLE")
print("="*70)
print(f"{'Model':<26}{'Train Acc':<12}{'Val Acc':<12}{'Val LogLoss':<14}{'Val Macro-F1':<14}")
print("-"*78)
print(f"{'Baseline (Elo only)':<26}{'-':<12}{baseline_acc:<12.3f}{'-':<14}{baseline_macro_f1:<14.3f}")
print(f"{'Final model (weighted HGB)':<26}{acc_tr:<12.3f}{acc_va:<12.3f}{ll_va:<14.3f}{macro_f1:<14.3f}")

n_draws_predicted = (preds_va == 'D').sum()
n_draws_correct = ((preds_va == 'D') & (y_val == 'D')).sum()
print(f"\nDraws predicted on validation: {n_draws_predicted}/20 | Correct: {n_draws_correct}/8 actual draws")

print()
print("Notes:")
print("- Val Acc / Macro-F1 measured on the 20 already-played 2026 World Cup matches")
print("  held out specifically as a realistic, recent, high-stakes test set.")
print("- Macro-F1 matters here because draws are a minority class (40% of validation,")
print("  23% of training) -- accuracy alone overstates performance on the classes")
print("  the model struggles with.")
print("- Train accuracy is NOT directly comparable to validation accuracy: train spans")
print("  150+ years of mixed competition types; validation is exclusively high-stakes")
print("  2026 World Cup matches between closely-matched qualified teams.")
print("- This model uses class weighting to balance draw/home/away prediction rates,")
print("  trading some raw accuracy for better balanced (macro-F1) performance versus")
print("  an unweighted model that nearly always ignores the draw class.")

# Save summary table to CSV
summary_df = pd.DataFrame([
    {'model': 'Baseline (Elo only)', 'train_acc': np.nan, 'val_acc': baseline_acc,
     'train_logloss': np.nan, 'val_logloss': np.nan, 'macro_f1': baseline_macro_f1, 'weighted_f1': np.nan},
    {'model': 'Final model (weighted HGB)', 'train_acc': acc_tr, 'val_acc': acc_va,
     'train_logloss': ll_tr, 'val_logloss': ll_va, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1},
])
summary_df.to_csv('model_summary_stats_final.csv', index=False)
print(f"\nSaved summary table to model_summary_stats_final.csv")
