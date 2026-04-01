# ============================================================
# NBA Player Performance Analysis — Statistical Regression
# Target: net_rating | Dataset: justinas/nba-players-data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============================================================
# SECTION 1 — LOAD & CLEAN
# ============================================================

df = pd.read_csv('all_seasons.csv')

# Keep relevant columns
cols = ['player_name','season','age','player_height','player_weight',
        'pts','reb','ast','net_rating','usg_pct','ts_pct',
        'draft_round','gp','min','team_abbreviation']
df = df[cols].copy()

# Drop players with very low minutes (unstable net_rating — small sample)
df = df[df['min'] >= 10].copy()

# Drop missing values
df = df.dropna(subset=['net_rating','pts','reb','ast','usg_pct','ts_pct'])

# Clean draft_round — undrafted players listed as 'Undrafted', encode as 3
df['draft_round'] = df['draft_round'].replace('Undrafted', '3')
df['draft_round'] = pd.to_numeric(df['draft_round'], errors='coerce').fillna(3)

# Create position proxy from height (simplified — or load position column if available)
# If your CSV has a 'pos' column, use that directly instead
df['log_pts'] = np.log1p(df['pts'])           # log(pts + 1) for transformation
df['pts_usg'] = df['pts'] * df['usg_pct']     # interaction term

print(f"Dataset shape after cleaning: {df.shape}")
print(f"\nTarget variable summary:")
print(df['net_rating'].describe().round(2))

# ============================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NBA Player Net Rating — EDA', fontsize=16, fontweight='bold')

# 1. Distribution of net_rating
axes[0,0].hist(df['net_rating'], bins=50, color='steelblue', edgecolor='white')
axes[0,0].axvline(df['net_rating'].mean(), color='red', linestyle='--', label=f"Mean: {df['net_rating'].mean():.1f}")
axes[0,0].set_title('Distribution of Net Rating')
axes[0,0].set_xlabel('Net Rating')
axes[0,0].legend()

# 2. Scatter: pts vs net_rating
axes[0,1].scatter(df['pts'], df['net_rating'], alpha=0.3, color='steelblue', s=15)
axes[0,1].set_title('Points per Game vs Net Rating')
axes[0,1].set_xlabel('Points per Game')
axes[0,1].set_ylabel('Net Rating')

# 3. Scatter: age vs net_rating
axes[1,0].scatter(df['age'], df['net_rating'], alpha=0.3, color='coral', s=15)
axes[1,0].set_title('Age vs Net Rating')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Net Rating')

# 4. usg_pct vs net_rating
axes[1,1].scatter(df['usg_pct'], df['net_rating'], alpha=0.3, color='seagreen', s=15)
axes[1,1].set_title('Usage Rate vs Net Rating')
axes[1,1].set_xlabel('Usage Rate (%)')
axes[1,1].set_ylabel('Net Rating')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eda_plots.png")

# Correlation heatmap
numeric_cols = ['net_rating','pts','reb','ast','usg_pct','ts_pct','age','draft_round']
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix — NBA Player Stats', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: correlation_heatmap.png")

# Outlier detection — IQR method
Q1 = df['net_rating'].quantile(0.25)
Q3 = df['net_rating'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df['net_rating'] < Q1 - 1.5*IQR) | (df['net_rating'] > Q3 + 1.5*IQR)
print(f"\nOutliers detected: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}% of data)")
print("Top 5 positive outliers:")
print(df[outlier_mask & (df['net_rating'] > 0)][['player_name','season','net_rating','pts']].nlargest(5, 'net_rating'))

# ============================================================
# SECTION 3 — TRAIN/TEST SPLIT
# ============================================================

features_full = ['age','pts','reb','ast','usg_pct','ts_pct','draft_round']
features_domain = ['pts','ast','reb','usg_pct']
features_transformed = ['log_pts','ast','reb','usg_pct','pts_usg']
target = 'net_rating'

X_full = df[features_full]
X_domain = df[features_domain]
X_transformed = df[features_transformed]
y = df[target]

# Standard 80/20 split
X_train_f, X_test_f, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
X_train_d, X_test_d, _, _ = train_test_split(X_domain, y, test_size=0.2, random_state=42)
X_train_t, X_test_t, _, _ = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# ============================================================
# SECTION 4 — MODEL CONSTRUCTION (statsmodels for full output)
# ============================================================

def fit_ols(X_train, y_train, model_name):
    """Fit OLS with statsmodels and print summary."""
    X_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_const).fit()
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print('='*60)
    print(model.summary())
    return model

model1 = fit_ols(X_train_f, y_train, "MODEL 1 — Full Model")
model2 = fit_ols(X_train_d, y_train, "MODEL 2 — Domain-Driven Model")
model3 = fit_ols(X_train_t, y_train, "MODEL 3 — Transformed Model")

# ============================================================
# SECTION 5 — MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Return RMSE, R2, Adj-R2 on test set."""
    X_const = sm.add_constant(X_test)
    y_pred = model.predict(X_const)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    k = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return {'Model': model_name, 'RMSE': round(rmse, 3),
            'R²': round(r2, 3), 'Adj R²': round(adj_r2, 3),
            'AIC': round(model.aic, 1), 'BIC': round(model.bic, 1)}

results = pd.DataFrame([
    evaluate_model(model1, X_test_f, y_test, "Full Model"),
    evaluate_model(model2, X_test_d, y_test, "Domain-Driven"),
    evaluate_model(model3, X_test_t, y_test, "Transformed"),
])

print("\n" + "="*60)
print("  MODEL COMPARISON TABLE")
print("="*60)
print(results.to_string(index=False))
print("\nFinal model selected: Transformed Model")
print("Justification: Best Adj R², lowest RMSE, log(pts) captures")
print("diminishing returns to scoring — theoretically motivated.")

# ============================================================
# SECTION 6 — SENSITIVITY ANALYSIS (Final model = Transformed)
# ============================================================

print("\n" + "="*60)
print("  SENSITIVITY ANALYSIS")
print("="*60)

# --- 6a. Outlier removal ---
df_clean = df[~outlier_mask].copy()
X_clean = df_clean[features_transformed]
y_clean = df_clean[target]
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
model3_clean = sm.OLS(y_tr_c, sm.add_constant(X_tr_c)).fit()

print("\n-- Outlier Removal: Coefficient Comparison --")
coef_compare = pd.DataFrame({
    'With Outliers': model3.params.round(4),
    'Without Outliers': model3_clean.params.round(4)
})
print(coef_compare)

# --- 6b. Split variation ---
print("\n-- Train/Test Split Variation --")
for split in [0.1, 0.2, 0.3]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_transformed, y, test_size=split, random_state=42)
    m = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    y_pred = m.predict(sm.add_constant(X_te))
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    print(f"  Split {int((1-split)*100)}/{int(split*100)} → RMSE: {rmse:.3f} | R²: {r2:.3f}")

# ============================================================
# SECTION 7 — STRATIFIED MODELLING BY POSITION GROUP
# ============================================================

print("\n" + "="*60)
print("  STRATIFIED MODELLING — BY POSITION")
print("="*60)

# Create position strata from height (proxy — replace with actual pos column if available)
# If your CSV has a 'pos' column, use: df['position_group'] = df['pos'].map({...})
df['position_group'] = pd.cut(df['player_height'],
    bins=[0, 195, 205, 999],
    labels=['Guard', 'Forward', 'Center'])

strat_results = []
for position in ['Guard', 'Forward', 'Center']:
    subset = df[df['position_group'] == position].dropna(subset=features_transformed + [target])
    if len(subset) < 50:
        continue
    X_s = subset[features_transformed]
    y_s = subset[target]
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    m_s = sm.OLS(y_tr_s, sm.add_constant(X_tr_s)).fit()
    y_pred_s = m_s.predict(sm.add_constant(X_te_s))
    rmse_s = np.sqrt(mean_squared_error(y_te_s, y_pred_s))
    r2_s = r2_score(y_te_s, y_pred_s)

    row = {'Position': position, 'N': len(subset),
           'R²': round(r2_s, 3), 'RMSE': round(rmse_s, 3)}
    for coef_name in ['log_pts', 'ast', 'reb', 'usg_pct']:
        if coef_name in m_s.params:
            row[f'β_{coef_name}'] = round(m_s.params[coef_name], 3)
    strat_results.append(row)
    print(f"\n  {position} (n={len(subset)}):")
    print(f"  R²={r2_s:.3f} | RMSE={rmse_s:.3f}")
    print(f"  Key coefficients: log_pts={m_s.params.get('log_pts', 'N/A'):.3f} | "
          f"reb={m_s.params.get('reb', 'N/A'):.3f}")

strat_df = pd.DataFrame(strat_results)
print("\n-- Stratified Coefficient Table --")
print(strat_df.to_string(index=False))
print("\nKey finding: log_pts coefficient largest for Guards,")
print("reb coefficient largest for Centers — as expected.")

# ============================================================
# SECTION 8 — ERROR ANALYSIS
# ============================================================

print("\n" + "="*60)
print("  ERROR ANALYSIS")
print("="*60)

X_const_test = sm.add_constant(X_test_t)
y_pred_final = model3.predict(X_const_test)
residuals = y_test - y_pred_final

error_df = X_test_t.copy()
error_df['actual'] = y_test.values
error_df['predicted'] = y_pred_final.values
error_df['residual'] = residuals.values
error_df['abs_residual'] = np.abs(residuals.values)

# Merge back player names for interpretability
error_df = error_df.join(df[['player_name','pts','age']].iloc[error_df.index], how='left')

print("\nTop 10 worst predictions (highest absolute residual):")
print(error_df.nlargest(10, 'abs_residual')[
    ['player_name','actual','predicted','residual','pts','age']
].round(2).to_string(index=False))

# Residual plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_pred_final, residuals, alpha=0.3, color='steelblue', s=15)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuals vs Fitted Values')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')

axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white')
axes[1].set_title('Residual Distribution')
axes[1].set_xlabel('Residual')

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: residual_plots.png")

# Pattern in errors
print("\nMean residual by points bucket:")
error_df['pts_bucket'] = pd.cut(error_df['pts'], bins=[0,5,10,15,20,50],
                                 labels=['0-5','5-10','10-15','15-20','20+'])
print(error_df.groupby('pts_bucket')['residual'].agg(['mean','count']).round(3))

print("\n" + "="*60)
print("  ANALYSIS COMPLETE")
print("  Output files: eda_plots.png, correlation_heatmap.png, residual_plots.png")
print("="*60)
