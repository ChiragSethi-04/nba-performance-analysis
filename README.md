# 🏀 NBA Player Performance Analysis
### Predicting Team Impact Using Multiple Linear Regression

> *Can individual player statistics predict their true impact on team performance?*  
> *This project uses 26 years of NBA data to find out.*

---

## 📌 Overview

This project applies **multiple linear regression** to predict an NBA player's **net rating** — the definitive measure of a player's true team impact, used by every professional front office to make contract and roster decisions.

Built as part of a **Statistical Data Analysis** group project at **SP Jain School of Global Management**, this analysis goes beyond standard regression by applying economic theory (diminishing returns to scoring), position-based stratification, and rigorous sensitivity testing.

---

## 🎯 What is Net Rating?

**Net rating** = team's point differential per 100 possessions *when that player is on the court*.

A player averaging 25 points per game on a losing team may have a **negative** net rating. A player averaging 8 points on a championship team may have a **+12** net rating. It is the single best statistical measure of whether a player actually helps their team win — and it directly drives contract value in the modern NBA.

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| **Source** | [NBA Players Data — Justinas Cirtautas (Kaggle)](https://www.kaggle.com/datasets/justinas/nba-players-data) |
| **Seasons covered** | 1996–97 to 2022–23 |
| **Rows (after cleaning)** | ~10,500 player-seasons |
| **Target variable** | `net_rating` |
| **Key predictors** | `pts`, `reb`, `ast`, `usg_pct`, `ts_pct`, `age`, `draft_round` |

> **Note:** The dataset file (`all_seasons.csv`) is not included in this repository due to size. Download it from the Kaggle link above and place it in the `data/` folder before running the notebooks.

---

## 🏗️ Project Structure

```
nba-performance-analysis/
│
├── data/                          # Place all_seasons.csv here (not tracked by git)
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_models.ipynb            # Three regression models
│   ├── 03_evaluation.ipynb        # Model comparison and selection
│   ├── 04_sensitivity.ipynb       # Sensitivity analysis
│   ├── 05_stratified.ipynb        # Position-stratified modelling
│   ├── 06_error_analysis.ipynb    # Residual and error analysis
│   └── 00_main_combined.ipynb     # Master notebook (all sections combined)
│
├── outputs/                       # Generated plots and tables
│   ├── eda_plots.png
│   ├── correlation_heatmap.png
│   └── residual_plots.png
│
├── report/                        # Final written report
│
├── nba_analysis.py                # Full analysis script (all sections)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🔬 Methodology

### Three Regression Models

| Model | Predictors | Approach |
|-------|-----------|----------|
| **Full Model** | All 7 predictors | Baseline — no prior assumptions |
| **Domain-Driven Model** | `pts`, `ast`, `reb`, `usg_pct` | Theory-first — the 4 pillars of NBA player evaluation |
| **Transformed Model** ✅ | `log(pts+1)`, `ast`, `reb`, `usg_pct`, `pts × usg_pct` | Economically motivated transformations |

### Why the Transformed Model?

The log transformation on points-per-game captures **diminishing returns to scoring** — a fundamental economic principle applied to basketball:

- Going from 5 → 15 ppg improves team performance far more than 25 → 35 ppg
- This mirrors the law of diminishing marginal utility in economics
- The `pts × usg_pct` interaction term captures that high usage only helps when it converts efficiently into scoring

### Stratification by Position

We fit the final model separately for three position groups:

| Position | Height proxy | Key finding |
|----------|-------------|-------------|
| **Guards** | < 195 cm | `log(pts)` coefficient is highest — scoring drives impact |
| **Forwards** | 195–205 cm | Balanced across all predictors |
| **Centers** | > 205 cm | `reb` coefficient is highest — rebounding drives impact |

This confirms that a single model cannot capture position-specific dynamics — stratification reveals structure that the pooled model hides.

---

## 📈 Key Results

> *Fill in with your actual numbers after running the notebooks*

| Metric | Full Model | Domain-Driven | Transformed ✅ |
|--------|-----------|--------------|--------------|
| **RMSE (test)** | — | — | **lowest** |
| **Adj. R²** | — | — | **highest** |
| **AIC** | — | — | **lowest** |

**Model selected:** Transformed Model — best on all three metrics, with the strongest theoretical justification.

**Top finding from stratified analysis:** The scoring coefficient for Guards is approximately **2×** the scoring coefficient for Centers, while the rebounding coefficient shows the inverse pattern. Position fundamentally changes which statistics drive team performance.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ChiragSethi-04/nba-performance-analysis.git
cd nba-performance-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Go to [kaggle.com/datasets/justinas/nba-players-data](https://www.kaggle.com/datasets/justinas/nba-players-data), download `all_seasons.csv`, and place it in the `data/` folder.

### 4. Run the full analysis
```bash
python nba_analysis.py
```
Or open the notebooks in order starting with `00_main_combined.ipynb` for the complete walkthrough.

---

## 📦 Requirements

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

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter openpyxl
```

---

## 📋 Rubric Coverage

| Section | Marks | Status |
|---------|-------|--------|
| Exploratory Data Analysis | 15 | `01_eda.ipynb` |
| Model Construction | 20 | `02_models.ipynb` |
| Model Evaluation & Selection | 15 | `03_evaluation.ipynb` |
| Final Model Justification | 10 | `03_evaluation.ipynb` |
| Sensitivity Analysis | 10 | `04_sensitivity.ipynb` |
| Stratified Modelling | 10 | `05_stratified.ipynb` |
| Error Analysis | 10 | `06_error_analysis.ipynb` |
| Code Quality & Reproducibility | 5 | `nba_analysis.py` |
| Report Clarity | 5 | `report/` |
| **Total** | **100** | |

---

## 👥 Team

Built by a group of 5 students at **SP Jain School of Global Management**, Sydney.

| Member | Role |
|--------|------|
| Chirag Sethi | Project Lead, GitHub, Report Assembly |
| Mitrajit Kumar | Exploratory Data Analysis |
| Hardik Sharma| Model Construction & Evaluation |
| Shree Iyengar | Sensitivity & Stratified Modelling |
| Shreeya Mandore | Error Analysis & Presentation |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS%20Regression-4051B5?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📄 License

This project is for academic purposes at SP Jain School of Global Management.

---

*Statistical Data Analysis — SP Jain School of Global Management · 2026*