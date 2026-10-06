# Project 2 — Codebook & Data Dictionary
## Startup Funding Prediction Model

**Document Purpose**: Definitive technical reference and variable codebook for Project 2 (Startup Funding Prediction Model). Documents all raw, longitudinal, and network bridge features derived from Project 1.

---

## 1. Dataset Overview & Scope

- **Sample Size**: 3,044 venture funding transactions across the Indian startup ecosystem.
- **Temporal Horizon**: January 2, 2015 to January 13, 2020.
- **Disclosed Amounts Sample**: 2,066 rounds with verified monetary amounts (range: $16,000 to $3,900,000,000; median: $1,725,000).
- **Stage Graduation Sample**: 3,044 rounds tracked across 2,349 distinct corporate entities.

---

## 2. Project 1 to Project 2 Network Bridge Specification

Rather than analyzing startup transactions in isolation, Project 2 directly leverages structural network metrics generated from Project 1's tripartite graph (4,002 nodes, 3,881 edges).

Participating investors in each transaction are indexed against Project 1:
- **Lead Investor PageRank (`lead_inv_pagerank`)**: Captures recursive capital prestige and elite fund influence.
- **Lead Investor Degree Centrality (`lead_inv_degree`)**: Quantifies direct deal volume and portfolio breadth.
- **Lead Investor Betweenness Centrality (`lead_inv_betweenness`)**: Measures information brokerage and syndication gatekeeping.
- **Lead Investor Eigenvector Centrality (`lead_inv_eigenvector`)**: Measures prestige by association with other central funds.
- **Syndicate Average PageRank (`syndicate_avg_pagerank`)**: Quantifies the collective prestige across all co-investors.
- **Marquee Investor Flag (`has_marquee_investor`)**: Binary indicator for tier-1 funds (e.g., Sequoia/Peak XV, Accel, Tiger Global, Blume, Matrix, Kalaari, Nexus, SAIF/Elevation).
- **Core Network Indicator (`in_p1_network`)**: Distinguishes startups backed by established venture institutions (1,487 rounds; 48.9%) from isolated angel rounds.

---

## 3. Variable Inventory & Data Dictionary

| Variable Name | Origin / Source | Data Type | Modeling Role | Preprocessing & Imputation | Definition & Formula |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `amount_usd` | Project 2 CSV | Float | Raw / Target | Cleaned numeric; non-positive/undisclosed set to NaN | Total capital raised in the financing round ($USD). |
| `log_amount_usd` | Engineered | Float | **Target A (Regression)** | $\log(\text{amount\_usd})$ for $\text{amount\_usd} > 0$ | Natural logarithm of round size. Symmetrical continuous target. |
| `graduated_next_stage` | Engineered | Binary (0/1) | **Target B (Classification)** | 1 if startup completed $\ge 2$ rounds, else 0 | Target measuring startup progression into follow-on funding. |
| `round_date` | Project 2 CSV | Datetime | Temporal Anchor | Regex normalized DD/MM/YYYY | Execution date of the term sheet. |
| `prev_funding_usd` | Engineered | Float | Predictor (Track Record) | Cumulative sum strictly prior to `round_date` | Total cumulative capital raised across prior rounds ($USD). |
| `log_prev_funding` | Engineered | Float | Predictor (Track Record) | $\log(1 + \text{prev\_funding\_usd})$ | Log-transformed prior cumulative capital. |
| `prev_round_size` | Engineered | Float | Predictor (Track Record) | Lagged amount of immediate prior round | Capital raised in the immediate preceding round ($USD; 0 for round 1). |
| `log_prev_round_size` | Engineered | Float | Predictor (Track Record) | $\log(1 + \text{prev\_round\_size})$ | Log-transformed preceding round amount. |
| `num_prev_rounds` | Engineered | Integer | Predictor (Track Record) | Running count of prior rounds | Count of completed financing events prior to current round. |
| `days_since_prev_round` | Engineered | Float | Predictor (Track Record) | Date delta in days; -1 for round 1 | Financing runway: days elapsed since preceding round. |
| `is_first_round` | Engineered | Binary (0/1) | Predictor (Track Record) | 1 if `num_prev_rounds == 0`, else 0 | Flag capturing inaugural financing transactions. |
| `startup_age_days` | Engineered | Float | Predictor (Operational) | Days elapsed from first recorded appearance | Operational age in days at the time of financing. |
| `sector_clean` | Project 2 CSV | Categorical | Predictor (Demographic) | One-hot encoded (8 tiers) | Primary industry: FinTech, E-Commerce, EdTech, Healthcare, Logistics, SaaS, Consumer, Food. |
| `hub_clean` | Project 2 CSV | Categorical | Predictor (Demographic) | One-hot encoded (8 tiers) | Ecosystem cluster: Bengaluru, Delhi-NCR, Mumbai, Pune, Hyderabad, Chennai, Jaipur, Tier-2. |
| `stage_clean` | Project 2 CSV | Categorical | Predictor (Demographic) | One-hot encoded (8 tiers) | Standardized round stage: Seed/Angel, Pre-Series A, Series A, Series B, Series C, Late Stage (C+), PE, Debt. |
| `num_investors` | Project 2 CSV | Integer | Predictor (Syndicate) | Count of comma-delimited entities | Number of participating investors in the round syndicate. |
| `lead_inv_degree` | Project 1 Graph | Float | **Predictor (Network Bridge)** | Max degree among participating funds | Normalized degree centrality of lead investor from Project 1. |
| `lead_inv_betweenness` | Project 1 Graph | Float | **Predictor (Network Bridge)** | Max betweenness among funds | Shortest-path betweenness centrality (brokerage power). |
| `lead_inv_eigenvector` | Project 1 Graph | Float | **Predictor (Network Bridge)** | Max eigenvector among funds | Eigenvector centrality (prestige by association). |
| `lead_inv_pagerank` | Project 1 Graph | Float | **Predictor (Network Bridge)** | Max PageRank among funds | PageRank prestige (recursive capital random-walk probability). |
| `syndicate_avg_pagerank` | Project 1 Graph | Float | **Predictor (Network Bridge)** | Mean PageRank across all funds | Average prestige of the entire investor syndicate. |
| `has_marquee_investor` | Project 1 Graph | Binary (0/1) | **Predictor (Network Bridge)** | 1 if tier-1 VC present, else 0 | Flag for Sequoia/Peak XV, Accel, Tiger Global, Blume, Matrix, Kalaari, Nexus, SAIF. |
| `in_p1_network` | Project 1 Graph | Binary (0/1) | **Predictor (Network Bridge)** | 1 if matched to Project 1, else 0 | Network coverage indicator distinguishing core funds from isolated angels. |
| `founder_team_size` | Project 1 Match | Float | Predictor (Founder) | Matched founder count; default 1.0 | Number of co-founders from Project 1 match. |
| `has_p1_founder_data` | Project 1 Match | Binary (0/1) | Predictor (Founder) | 1 if founder named, else 0 | Flag indicating whether verified founder team data is present. |

---

## 4. Modeling Protocols & Benchmarks

### Regression Models (Target: `log_amount_usd`)
- OLS Linear Regression: $R^2 = 0.639$, RMSE = 1.184
- Ridge Regression ($\alpha=1.0$): $R^2 = 0.640$, RMSE = 1.182
- Lasso Regression ($\alpha=0.01$): $R^2 = 0.641$, RMSE = 1.181
- ElasticNet ($\alpha=0.01, \text{l1\_ratio}=0.5$): $R^2 = 0.641$, RMSE = 1.181
- Random Forest Regressor: $R^2 = 0.686$, RMSE = 1.105
- **XGBoost Regressor**: **$R^2 = 0.698$, RMSE = 1.084, MAE = 0.840**

### Classification Models (Target: `graduated_next_stage`)
- **Logistic Regression**: **ROC-AUC = 0.890**, Accuracy = 85.1%, F1 = 0.771
- **XGBoost Classifier**: **ROC-AUC = 0.877**, Accuracy = 85.4%, Precision = 95.7%, F1 = 0.777
- Random Forest Classifier: ROC-AUC = 0.873, Accuracy = 85.7%, Precision = 98.7%, F1 = 0.777

### Ablation Findings
Inclusion of Project 1 network centrality features generates a statistically significant performance boost across every model (+5.5% $R^2$ increase for tree ensembles, +3.3% $R^2$ for linear specifications), demonstrating that network position acts as an independent certification mechanism in the Indian venture capital market.
