"""
train.py — LifeSnaps Calories Regression
15 features (XGBoost 95%-importance from nb01, minus high-missing cols from nb02).
8 models tuned with GroupKFold, then evaluated across 7 strategies.
Usage: python train.py
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import cross_val_score, cross_validate, GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from lifesnaps_lib.preprocessing import PerUserMedianImputer

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, 'rais_anonymized', 'csv_rais_anonymized',
                          'daily_fitbit_sema_df_unprocessed.csv')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  [plot] saved → {path}')


# ════════════════════════════════════════════════════════════════════════════
# 1. Load & Filter Data
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 1. Load Data ===')
df_raw = pd.read_csv(DATA_PATH, parse_dates=['date'])
print(f'Raw shape: {df_raw.shape}')
print(f'Users: {df_raw["id"].nunique()}')
print(f'Date range: {df_raw["date"].min()} → {df_raw["date"].max()}')

df = df_raw.dropna(subset=['calories']).copy()
df = df[df['calories'] >= 500].copy()
df = df.sort_values(['id', 'date']).reset_index(drop=True)
print(f'After filtering: {df.shape}  |  Users: {df["id"].nunique()}')


# ════════════════════════════════════════════════════════════════════════════
# 2. Feature Engineering  (15 features — nb02 conclusion)
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 2. Feature Engineering ===')
df['TotalActiveMinutes'] = (df['very_active_minutes']
                            + df['moderately_active_minutes']
                            + df['lightly_active_minutes'])
df['ActiveRatio'] = df['TotalActiveMinutes'] / (
    df['TotalActiveMinutes'] + df['sedentary_minutes'] + 1e-9)
df['StepsPerActiveMin'] = np.where(
    df['TotalActiveMinutes'] > 0, df['steps'] / df['TotalActiveMinutes'], 0)
df['hr_zone_cardio'] = df['minutes_in_default_zone_2']

# 15 features: 19 from nb01 XGBoost-95% importance, minus 4 high-missing cols
# dropped: nremhr, rmssd, ALERT, TIRED  (>50% missing, MAE diff only +1 kcal)
feature_cols = [
    # Activity (raw)
    'steps', 'distance', 'very_active_minutes', 'moderately_active_minutes',
    'lightly_active_minutes',
    # Activity (engineered)
    'TotalActiveMinutes', 'ActiveRatio', 'StepsPerActiveMin',
    # Heart rate
    'bpm', 'resting_hr', 'hr_zone_cardio',
    # Advanced Fitbit
    'filteredDemographicVO2Max',
    # Skin temperature
    'nightly_temperature', 'daily_temperature_variation',
    # Demographics
    'bmi',
]
assert len(feature_cols) == 15, f'Expected 15 features, got {len(feature_cols)}'

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

target = 'calories'
X      = df[['id'] + feature_cols]
y      = df[target]
groups = df['id']
print(f'Features: {len(feature_cols)}  |  X shape: {X.shape}')


# ════════════════════════════════════════════════════════════════════════════
# 3. Target Distribution
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 3. Target Distribution ===')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=40, edgecolor='white', color='#2E75B6', alpha=0.85)
axes[0].axvline(y.mean(),   color='red',    linestyle='--', label=f'Mean: {y.mean():.0f}')
axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Median: {y.median():.0f}')
axes[0].set_title('Calories Distribution', fontweight='bold')
axes[0].set_xlabel('Calories (kcal)')
axes[0].legend()

rows_per_user = df.groupby('id').size().sort_values()
axes[1].bar(range(len(rows_per_user)), rows_per_user.values, color='#ED7D31', alpha=0.85)
axes[1].axhline(rows_per_user.mean(), color='red', linestyle='--',
                label=f'Mean: {rows_per_user.mean():.0f} days')
axes[1].set_title('Days per User', fontweight='bold')
axes[1].set_xlabel('User (sorted)')
axes[1].set_ylabel('Number of days')
axes[1].legend()
savefig('target_distribution.png')
print(f'Calories — mean: {y.mean():.0f}, std: {y.std():.0f}, '
      f'min: {y.min():.0f}, max: {y.max():.0f}')


# ════════════════════════════════════════════════════════════════════════════
# 4. Models & Pipeline
# ════════════════════════════════════════════════════════════════════════════
def make_pipe(model):
    return Pipeline([
        ('imputer', PerUserMedianImputer(feature_cols=feature_cols)),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])

default_models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'Lasso':             Lasso(alpha=1.0, max_iter=5000),
    'ElasticNet':        ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=2),
    'LightGBM':          lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=2),
}


# ════════════════════════════════════════════════════════════════════════════
# 5. Optuna Tuning — GroupKFold  (the only tuning pass)
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 5. Optuna Tuning (GroupKFold) ===')
gkf      = GroupKFold(n_splits=3)
N_TRIALS = 20

def model_ridge(trial):
    return Ridge(alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True))

def model_lasso(trial):
    return Lasso(alpha=trial.suggest_float('alpha', 1e-3, 1e2, log=True), max_iter=5000)

def model_elasticnet(trial):
    return ElasticNet(alpha=trial.suggest_float('alpha', 1e-3, 1e2, log=True),
                      l1_ratio=trial.suggest_float('l1_ratio', 0.01, 0.99), max_iter=5000)

def model_rf(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 400),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        random_state=42, n_jobs=-1)

def model_gb(trial):
    return GradientBoostingRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        max_depth=trial.suggest_int('max_depth', 2, 6),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        random_state=42)

def model_xgb(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        max_depth=trial.suggest_int('max_depth', 2, 8),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 5.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1.0, 10.0),
        random_state=42, verbosity=0, n_jobs=2)

def model_lgbm(trial):
    return lgb.LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        max_depth=trial.suggest_int('max_depth', 2, 8),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 5.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1.0, 10.0),
        random_state=42, verbose=-1, n_jobs=2)

model_fns = {
    'Ridge': model_ridge, 'Lasso': model_lasso, 'ElasticNet': model_elasticnet,
    'Random Forest': model_rf, 'Gradient Boosting': model_gb,
    'XGBoost': model_xgb, 'LightGBM': model_lgbm,
}

studies = {}
for name, fn in model_fns.items():
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial, fn=fn: cross_val_score(
            make_pipe(fn(trial)), X, y, cv=gkf, groups=groups,
            scoring='neg_mean_absolute_error', n_jobs=2).mean(),
        n_trials=N_TRIALS, n_jobs=1)
    studies[name] = study
    print(f'  {name:<22} GKF-tuned MAE={-study.best_value:.1f}')

def get_tuned_model(name):
    if name == 'Linear Regression':
        return LinearRegression()
    p = studies[name].best_params
    if name == 'Ridge':             return Ridge(alpha=p['alpha'])
    if name == 'Lasso':             return Lasso(alpha=p['alpha'], max_iter=5000)
    if name == 'ElasticNet':        return ElasticNet(alpha=p['alpha'], l1_ratio=p['l1_ratio'], max_iter=5000)
    if name == 'Random Forest':     return RandomForestRegressor(**p, random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting': return GradientBoostingRegressor(**p, random_state=42)
    if name == 'XGBoost':           return XGBRegressor(**p, random_state=42, verbosity=0, n_jobs=2)
    if name == 'LightGBM':          return lgb.LGBMRegressor(**p, random_state=42, verbose=-1, n_jobs=2)


# ════════════════════════════════════════════════════════════════════════════
# 6. Evaluate all 8 models across 7 strategies
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 6. Model Comparison across Strategies ===')

tscv        = TimeSeriesSplit(n_splits=3)
N_SPLITS    = 3
MIN_TRAIN   = 15
MIN_TEST    = 5
tscv_user   = TimeSeriesSplit(n_splits=N_SPLITS)

df_sorted   = df.sort_values('date').reset_index(drop=True)
X_sorted    = df_sorted[['id'] + feature_cols]
y_sorted    = df_sorted[target]

df_sorted2  = df.sort_values(['id', 'date']).reset_index(drop=True)
X_wu        = df_sorted2[['id'] + feature_cols]
y_wu        = df_sorted2[target]

unique_users = df['id'].unique()

all_results = []

for name in default_models:
    model = get_tuned_model(name)
    row   = {'Model': name}

    # ── GroupKFold ──
    cv = cross_validate(make_pipe(model), X, y, cv=gkf, groups=groups, n_jobs=2,
                        scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'})
    row['GKF MAE'] = -cv['test_mae'].mean()
    row['GKF R²']  =  cv['test_r2'].mean()

    # ── TSCV ──
    cv = cross_validate(make_pipe(model), X_sorted, y_sorted, cv=tscv, n_jobs=2,
                        scoring={'mae': 'neg_mean_absolute_error'})
    row['TSCV MAE'] = -cv['test_mae'].mean()

    # ── WU-Single (Within-User, last-20% holdout) ──
    train_idx, test_idx = [], []
    for uid, grp in df_sorted2.groupby('id'):
        n = len(grp); n_test = max(1, int(n * 0.2))
        train_idx.extend(grp.index[:-n_test].tolist())
        test_idx.extend(grp.index[-n_test:].tolist())
    pipe = make_pipe(model)
    pipe.fit(X_wu.iloc[train_idx], y_wu.iloc[train_idx])
    row['WU-Single MAE'] = mean_absolute_error(y_wu.iloc[test_idx],
                                                pipe.predict(X_wu.iloc[test_idx]))

    # ── WU-TSCV ──
    df_s = df.sort_values('date').reset_index(drop=True)
    X_a  = df_s[['id'] + feature_cols]; y_a = df_s[target]
    fold_maes = []
    for tr, te in tscv_user.split(df_s):
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        p = make_pipe(model)
        p.fit(X_a.iloc[tr], y_a.iloc[tr])
        fold_maes.append(mean_absolute_error(y_a.iloc[te], p.predict(X_a.iloc[te])))
    row['WU-TSCV MAE'] = float(np.mean(fold_maes))

    # ── PU-Single (Per-User, last-20% holdout per user) ──
    user_maes = []
    for uid, grp in df.sort_values('date').groupby('id'):
        grp = grp.reset_index(drop=True)
        n = len(grp); n_test = max(1, int(n * 0.2))
        if n - n_test < 10:
            continue
        X_u = grp[['id'] + feature_cols]; y_u = grp[target]
        p = make_pipe(model)
        p.fit(X_u.iloc[:-n_test], y_u.iloc[:-n_test])
        user_maes.append(mean_absolute_error(y_u.iloc[-n_test:],
                                              p.predict(X_u.iloc[-n_test:])))
    row['PU-Single MAE'] = float(np.mean(user_maes))

    # ── PU-TSCV ──
    fold_maes = []
    for uid, grp in df.sort_values('date').groupby('id'):
        grp = grp.reset_index(drop=True)
        X_u = grp[['id'] + feature_cols]; y_u = grp[target]
        for tr, te in tscv_user.split(grp):
            if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
                continue
            p = make_pipe(model)
            p.fit(X_u.iloc[tr], y_u.iloc[tr])
            fold_maes.append(mean_absolute_error(y_u.iloc[te], p.predict(X_u.iloc[te])))
    row['PU-TSCV MAE'] = float(np.mean(fold_maes))

    # ── LOSO ──
    loso_maes = []
    for test_user in unique_users:
        tr_mask = df['id'] != test_user
        te_mask = df['id'] == test_user
        if te_mask.sum() == 0:
            continue
        p = make_pipe(model)
        p.fit(X[tr_mask], y[tr_mask])
        loso_maes.append(mean_absolute_error(y[te_mask], p.predict(X[te_mask])))
    row['LOSO MAE'] = float(np.mean(loso_maes))

    all_results.append(row)
    print(f'  {name:<22} GKF={row["GKF MAE"]:.1f}  TSCV={row["TSCV MAE"]:.1f}  '
          f'WU-S={row["WU-Single MAE"]:.1f}  WU-T={row["WU-TSCV MAE"]:.1f}  '
          f'PU-S={row["PU-Single MAE"]:.1f}  PU-T={row["PU-TSCV MAE"]:.1f}  '
          f'LOSO={row["LOSO MAE"]:.1f}')

results_df = pd.DataFrame(all_results).set_index('Model')
mae_cols   = ['GKF MAE', 'TSCV MAE', 'WU-Single MAE', 'WU-TSCV MAE',
              'PU-Single MAE', 'PU-TSCV MAE', 'LOSO MAE']
print('\nFull comparison table:')
print(results_df[mae_cols].round(1).to_string())

# Save all strategy results
results_path = os.path.join(MODELS_DIR, 'results_15features.csv')
results_df[mae_cols + ['GKF R²']].round(2).to_csv(results_path)
print(f'  [saved] results → {results_path}')


# ════════════════════════════════════════════════════════════════════════════
# 7. Comparison Plot
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 7. Comparison Plot ===')
strategy_labels = ['GKF', 'TSCV', 'WU-Single', 'WU-TSCV', 'PU-Single', 'PU-TSCV', 'LOSO']
colors = ['#9DC3E6', '#ED7D31', '#70AD47', '#9966CC', '#7030A0', '#FF9999', '#404040']

fig, ax = plt.subplots(figsize=(20, 7))
x     = np.arange(len(results_df))
width = 0.12
for i, (strat, col, color) in enumerate(zip(strategy_labels, mae_cols, colors)):
    offset = (i - len(strategy_labels) / 2 + 0.5) * width
    b = ax.bar(x + offset, results_df[col], width, label=strat, color=color, alpha=0.9)
    ax.bar_label(b, fmt='%.0f', padding=2, fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=15, ha='right')
ax.set_ylabel('MAE (kcal)')
ax.set_title('Model Comparison — 15 Features, 7 Evaluation Strategies', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
savefig('model_comparison.png')


# ════════════════════════════════════════════════════════════════════════════
# 8. Feature Importance
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 8. Feature Importance ===')
importance_dict = {}
for name in default_models:
    pipe = make_pipe(get_tuned_model(name))
    pipe.fit(X, y)
    model = pipe.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        scores = model.feature_importances_
    elif hasattr(model, 'coef_'):
        scores = np.abs(model.coef_)
    else:
        continue
    total = scores.sum()
    importance_dict[name] = pd.Series(
        scores / total if total > 0 else scores, index=feature_cols)

importance_df = pd.DataFrame(importance_dict)
importance_df['mean'] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(importance_df.drop(columns='mean'), cmap='YlOrRd',
            annot=True, fmt='.3f', linewidths=0.4, ax=ax)
ax.set_title('Feature Importance — All Models (15 features)', fontweight='bold')
savefig('feature_importance_heatmap.png')


# ════════════════════════════════════════════════════════════════════════════
# 9. Save Models
# ════════════════════════════════════════════════════════════════════════════
print('\n=== 9. Save Models ===')
saved_info = {}
for name in default_models:
    pipe = make_pipe(get_tuned_model(name))
    pipe.fit(X, y)
    safe_name = name.lower().replace(' ', '_')
    filepath  = os.path.join(MODELS_DIR, f'{safe_name}_gkf_tuned.joblib')
    joblib.dump(pipe, filepath)
    saved_info[name] = {
        'file':    filepath,
        'gkf_mae': round(float(results_df.loc[name, 'GKF MAE']), 2),
        'gkf_r2':  round(float(results_df.loc[name, 'GKF R²']), 4),
    }
    print(f'  Saved: {filepath}')

meta = {
    'feature_cols':  feature_cols,
    'n_features':    len(feature_cols),
    'target':        'calories',
    'n_samples':     int(len(y)),
    'n_users':       int(df['id'].nunique()),
    'tuning':        'Optuna GroupKFold (3 splits, 20 trials)',
    'models':        saved_info,
}
meta_path = os.path.join(MODELS_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'  Saved metadata: {meta_path}')

print('\n=== Done! ===')
print(f'  Plots  → {PLOTS_DIR}/')
print(f'  Models → {MODELS_DIR}/')
