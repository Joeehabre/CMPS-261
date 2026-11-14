# NASA Kepler Exoplanet Candidate Classification

**Course:** CMPS 261 Machine Learning (American University of Beirut)  
**Track:** Track A, 4.1 Tabular (NASA Kepler Objects of Interest)  
**Group 4:** Joe Habre and Joy Habib Najem  

We formulate NASA Kepler Objects of Interest (KOI) vetting ($m = 9,564$ observations across $8,214$ unique host stars and $153$ archive attributes) as a binary classification task distinguishing **Potential Exoplanets** (`CONFIRMED` $\cup$ `CANDIDATE`, $y = 1$, $4,725$ instances, $49.40\%$) from **False Positives** (`FALSE POSITIVE`, $y = 0$, $4,839$ instances, $50.60\%$).

## Repository Structure & Downloadable Model Weights

| Path | Description |
| :--- | :--- |
| `data/cumulative.csv` | Committed NASA Exoplanet Archive KOI cumulative dataset snapshot ($9,564 \times 153$) |
| `notebooks/02_final_exoplanet_classification.ipynb` | Final executed notebook: preprocessing, tuning, 5-fold grouped CV, ResNet-MLP, FT-Transformer, disjoint calibration, ablations, failure analysis, and locked test evaluation |
| `models/final_xgboost_model.joblib` | Downloadable trained final XGBoost classifier (`500` trees, `max_depth=6`, `lr=0.03`) |
| `models/final_resnet_mlp.pt` | Downloadable trained ResNet-MLP weights (`68,993` params) and disjoint temperature $T^* = 1.074$ |
| `models/final_ft_transformer.pt` | Downloadable trained FT-Transformer weights (`30,593` params) and disjoint temperature $T^* = 1.064$ |
| `models/final_neural_preprocessor.joblib` | Fitted median imputer (`add_indicator=True`) and `StandardScaler` pipeline for neural inference |
| `figures/fig1_roc_pr_curves.png` | Held-out test set ROC and Precision-Recall curves ($300$ DPI) |
| `figures/fig2_diagnostics_calibration.png` | Held-out test set Reliability Diagram and Complexity vs. Performance trade-off plot ($300$ DPI) |
| `Reports/Final_Report.pdf` | Final 7-section project report |
| `Reports/Poster.pdf` | Final 7-component project poster |
| `requirements.txt` | Reproducible Python dependencies |

## Summary of Final Results (Held-Out Test Set, $m_{\text{test}} = 1,902$, $0$ Shared Host Stars)

| Model | Capacity / Parameters | 5-Fold Grouped CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test ECE ($\downarrow$) | Test Brier ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Tuned $L_2, C=1.0$)** | $210$ weights | $0.9523 \pm 0.0073$ | $0.9525$ | $0.9475$ | $0.8686$ | $0.0376$ | $0.0868$ |
| **Random Forest ($500$ trees, depth$=16$)** | $360,068$ nodes | $0.9684 \pm 0.0045$ | $0.9665$ | $0.9671$ | $0.8933$ | $0.0413$ | $0.0742$ |
| **XGBoost (Tuned, depth$=6$, lr$=0.03$)** | $500$ boosted trees | $0.9711 \pm 0.0032$ | $0.9707$ | $0.9699$ | **0.9041** | $0.0189$ | $0.0677$ |
| **ResNet-MLP (Disjoint Temp. Scaled)** | $68,993$ params | $0.9621 \pm 0.0044$ | $0.9627$ | $0.9629$ | $0.8764$ | $0.0258$ | $0.0776$ |
| **FT-Transformer (Disjoint Temp. Scaled)** | $30,593$ params | $0.9653 \pm 0.0047$ | $0.9626$ | $0.9625$ | $0.8808$ | $0.0208$ | $0.0778$ |
| **Calibrated Ensemble ($0.8, 0.1, 0.1$)** | Trees + $99,586$ params | **0.9715 $\pm$ 0.0036** | **0.9709** | **0.9703** | **0.9041** | **0.0157** | **0.0669** |