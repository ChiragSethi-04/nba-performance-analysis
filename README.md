# NBA Player Performance Analysis
### Predicting Team Impact from Individual Statistics Using Multiple Linear Regression

> *26 seasons. 10,720 player-seasons. One question: can what a player does on the court predict whether their team wins?*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS%20Regression-4051B5?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## The Problem

NBA front offices spend hundreds of millions of dollars on player contracts. The central question they face every offseason: *which player statistics actually predict team-level winning?*

**Net rating** — a team's point differential per 100 possessions with a given player on the court — is the industry-standard answer. A player averaging 25 points on a losing team may post a −5 net rating. A role player on a championship team may post +12. It is the single number that separates the scorers from the winners.

This project builds and stress-tests a regression model that predicts net rating from individual player statistics, applies economic theory to improve it, and stratifies it by position to reveal structure that a pooled model hides.

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | [NBA Players Dataset — Justinas Cirtautas (Kaggle)](https://www.kaggle.com/datasets/justinas/nba-players-data) |
| Seasons | 1996–97 to 2022–23 (26 seasons) |
| Raw rows | 12,844 player-seasons |
| After cleaning (`gp ≥ 20`, no nulls) | **10,720 player-seasons** |
| Target variable | `net_rating` (mean: −1.09, std: 6.44, range: −40 to +19.5) |
| Train / test split | 80% / 20%, `random_state=42` |

> `all_seasons.csv` is not committed due to size. Download from the Kaggle link above and place in `data/`.

---

## Analytical Approach

### Three Competing Models

| | Model 1 — Full | Model 2 — Domain-Driven | Model 3 — Transformed ✅ |
|--|----------------|------------------------|--------------------------|
| **Predictors** | `age`, `pts`, `reb`, `ast`, `usg_pct`, `ts_pct`, `draft_round` | `pts`, `ast`, `reb`, `usg_pct` | `log(pts+1)`, `ast`, `reb`, `usg_pct`, `pts × usg_pct` |
| **Motivation** | Baseline — all available stats | Theory-first — the 4 pillars of player evaluation | Economically motivated transformation |
| **Condition No.** | 1,140 ⚠️ | 407 | **242** ✓ |

### Why Model 3?

The log transformation on points-per-game applies a well-established economic principle — **diminishing marginal returns to scoring**:

- A player improving from 5 to 15 ppg contributes far more to winning than one improving from 25 to 35 ppg
- The `pts × usg_pct` interaction term captures that high usage only helps when it converts efficiently into scoring — high usage without payoff is structurally costly

Model 1 achieves the highest raw R², but its condition number of 1,140 signals serious multicollinearity, making its coefficients unreliable. Model 3 sacrifices 0.194 RMSE points for stable, interpretable, theoretically-grounded coefficients.

---

## Results

### Model Comparison (Test Set)

| Model | RMSE | R² | Adj R² | AIC | BIC |
|-------|------|----|--------|-----|-----|
| Full Model | 5.822 | 0.209 | 0.207 | 54,256.6 | 54,313.1 |
| Domain-Driven | 6.044 | 0.148 | 0.147 | 55,070.7 | 55,106.0 |
| **Transformed ✅** | **6.016** | **0.156** | **0.154** | **55,033.4** | **55,075.7** |

### Prediction Accuracy by Tolerance Band

| Model | Within ±1 | Within ±3 | Within ±5 |
|-------|-----------|-----------|-----------|
| Full Model | 13.9% | 40.4% | 62.2% |
| Domain-Driven | 14.1% | 39.5% | 61.2% |
| **Transformed** | **13.9%** | **39.1%** | **61.7%** |

A ±3 net-rating prediction is sufficient for a team to correctly classify a player as positive-impact, neutral, or negative-impact — the threshold that matters for contract evaluation.

### Model 3 — Final Coefficients

| Predictor | β | p-value | Interpretation |
|-----------|---|---------|----------------|
| `log_pts` | +2.51 | <0.001 | Scoring helps, but with diminishing returns |
| `ast` | +0.25 | <0.001 | Each assist/game adds +0.25 net rating |
| `reb` | +0.15 | <0.001 | Positive but modest contribution |
| `usg_pct` | −46.97 | <0.001 | High usage without payoff costs the team |
| `pts_usg` | +0.94 | <0.001 | High scorers who earn their usage offset the penalty |

**Intercept p = 0.192** — not significant, meaning all predictors are centred meaningfully around zero.

---

## Sensitivity Analysis

Three robustness tests were applied to validate that Model 3's conclusions are not artefacts of specific data choices.

### Test 1 — Outlier Removal (Univariate IQR)

112 outliers removed (1.04% of data). Coefficient stability:

| Predictor | Baseline β | After Removal β | % Change | Verdict |
|-----------|------------|-----------------|----------|---------|
| `reb` | 0.1512 | 0.1498 | −0.9% | **Robust** ✓ |
| `usg_pct` | −46.9703 | −45.5109 | +3.1% | **Robust** ✓ |
| `pts_usg` | 0.9403 | 1.0464 | +11.3% | Moderate |
| `ast` | 0.2503 | 0.2013 | −19.6% | Borderline |
| `log_pts` | 2.5105 | 1.9811 | −21.1% | Sensitive |

### Test 2 — Split Variation (5 Random Seeds)

| Seed | RMSE | R² |
|------|------|----|
| 42 | 6.016 | 0.156 |
| 123 | 6.108 | 0.128 |
| 7 | 5.995 | 0.118 |
| 99 | 6.107 | 0.111 |
| 2024 | 5.822 | 0.153 |

**RMSE range: 5.822 – 6.108** (spread of 0.286 net-rating points — negligible across a −40 to +19.5 outcome range). No predictor switches sign or loses significance in any split.

### Test 3 — Multi-dimensional IQR (4 variables simultaneously)

942 rows removed (8.8%). `log_pts` stability improved dramatically vs Test 1 (−6.0% vs −21.1%), confirming its Test 1 sensitivity was driven by a specific cluster of high-scoring, high-team-quality outliers — not random noise. **`usg_pct` remained robust across all three tests** — its negative direction is the most structurally consistent finding in the entire analysis.

---

## Stratified Modelling by Position

Positions assigned using player height as a proxy: Guards < 196 cm, Forwards 196–208 cm, Centers > 208 cm.

### Model Fit by Position

| Position | N | R² | Adj R² |
|----------|---|----|--------|
| Guard | 3,524 | 0.130 | 0.128 |
| Forward | 4,321 | 0.122 | 0.121 |
| **Center** | **2,875** | **0.170** | **0.169** |

Centers achieve the highest R² — their constrained, well-defined role means individual statistics translate most directly into net rating outcomes.

### Coefficient Comparison by Position

| Predictor | Guard β | Forward β | Center β | Key finding |
|-----------|---------|-----------|----------|-------------|
| `log_pts` | 1.89 *** | 3.04 *** | **3.92 ****** | Scoring impact **doubles** from Guards to Centers |
| `ast` | 0.18 ** | **0.58 ****** | 0.41 ** | Playmaking premium for non-traditional positions |
| `reb` | **0.27 *** (sig)** | −0.06 ✗ (ns) | −0.04 ✗ (ns) | Only significant for Guards |
| `usg_pct` | −46.92 | −50.17 | **−54.06** | Penalty escalates with position size |
| `pts_usg` | 1.05 | 0.73 | 1.13 | Strongest payoff for positional extremes |

**Three findings a pooled model cannot show:**

1. **`reb` is significant only for Guards** (p = 0.031). For Forwards and Centers, it is not (p = 0.270, p = 0.585). A single pooled coefficient of 0.15 completely hides this reversal — `reb` is the only predictor whose effect crosses zero between strata.

2. **`log_pts` rises monotonically** from 1.89 (Guards) to 3.92 (Centers) — a high-scoring Center is a strong indicator of system health because Centers only score heavily on well-constructed teams. For Guards, high scoring is baseline expectation and carries less marginal team-level information.

3. **`usg_pct` penalty escalates with position size** — high-usage Centers signal a team forced to run a slow post-dependent offense, the least flexible system. The gradient Guard → Forward → Center is consistent and monotonic.

---

## Regression to the Mean

All three models show meaningful prediction shrinkage — a direct consequence of R² < 1. The model cannot explain all variance, so predictions are compressed toward the mean of actuals (≈ −1.1).

| Model | Std Actuals | Std Predictions | Shrinkage % |
|-------|-------------|-----------------|-------------|
| Full Model | 6.44 | ~4.18 | ~35% |
| Domain-Driven | 6.44 | ~3.78 | ~41% |
| Transformed | 6.44 | ~3.77 | ~41% |

The Full Model shrinks least (highest R²). Models 2 and 3 shrink more, meaning extreme players — those with very high or very low true net ratings — are systematically underpredicted and overpredicted respectively. This is expected OLS behaviour when predictors explain roughly 15–20% of outcome variance.

---

## Visualisations

All outputs saved to `outputs/`:

| File | Description |
|------|-------------|
| `01_net_rating_distribution.png` | Distribution of net_rating across all player-seasons |
| `02_correlation_heatmap.png` | Predictor correlation matrix |
| `03_scatter_plots.png` | Key predictor vs net_rating scatter plots |
| `04_net_rating_by_draft_round.png` | Net rating distribution by draft round |
| `05_model_comparison_chart.png` | RMSE and R² bar chart — all three models |
| `06_stratified_coef_comparison.png` | Grouped bar chart of coefficients by position |
| `07_residuals_qq.png` | Residuals vs fitted + QQ plots for all three models |
| `08_actual_vs_predicted_strata.png` | Actual vs predicted scatter by position group |
| `09_coefficient_profile.png` | Line plot tracing each predictor's β across strata |
| `regression_to_mean.png` | Model 3 actual vs predicted with shrinkage annotation |

---

## Project Structure

```
nba-performance-analysis/
├── data/                            # Place all_seasons.csv here (not tracked)
├── notebooks/
│   ├── 01_EDA_ANALYSIS.ipynb        # Distributions, correlations, outlier detection
│   ├── 02_models.ipynb              # OLS model construction + coefficient analysis
│   ├── 03_Evaluation.ipynb          # Test-set evaluation, model selection, shrinkage
│   ├── 04_Sensitivity_Analysis.ipynb# Three robustness tests
│   ├── 05_Stratified.ipynb          # Position-stratified modelling
│   └── 06_Error_Analysis.ipynb      # Residual diagnostics
├── outputs/                         # All generated plots (10 files)
├── report/                          # Final written report
├── nba_analysis.py                  # Full analysis script (all sections combined)
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/ChiragSethi-04/nba-performance-analysis.git
cd nba-performance-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# → kaggle.com/datasets/justinas/nba-players-data
# → place all_seasons.csv in data/

# 4a. Run full analysis script
python nba_analysis.py

# 4b. Or step through notebooks in order (recommended for full output)
jupyter notebook
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
jupyter>=1.0.0
openpyxl>=3.0.0
```

---

## Team

Built for **Statistical Data Analysis** at **SP Jain School of Global Management**, Sydney.

| Member | Contribution |
|--------|-------------|
| Chirag Sethi | Project lead, GitHub, model evaluation, sensitivity analysis, report assembly |
| Mitrajit Kumar | Exploratory data analysis |
| Hardik Sharma | Model construction and OLS diagnostics |
| Shree Iyengar | Sensitivity analysis and stratified modelling |
| Shreeya Mandore | Error analysis and presentation |

---

*Statistical Data Analysis — SP Jain School of Global Management · 2026*
