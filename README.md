# NASA Kepler Exoplanet Candidate Classification

**Course:** CMPS 261 Machine Learning (American University of Beirut)  
**Track:** Track A, 4.1 Tabular (NASA Kepler Objects of Interest)  
**Group 4:** Joe Habre and Joy Habib Najem  

We formulate NASA Kepler Objects of Interest (KOI) vetting ($m = 9,564$ observations across $8,214$ unique host stars and $153$ archive attributes) as a binary classification task distinguishing **Potential Exoplanets** (`CONFIRMED` $\cup$ `CANDIDATE`, $y = 1$, $4,725$ instances, $49.40\%$) from **False Positives** (`FALSE POSITIVE`, $y = 0$, $4,839$ instances, $50.60\%$).

## Repository Structure

| Path | Description |
| :--- | :--- |
| `data/cumulative.csv` | Committed NASA Exoplanet Archive KOI cumulative dataset snapshot ($9,564 \times 153$) |
| `notebooks/01_milestone_baseline.ipynb` | Progress report milestone baseline notebook |
| `notebooks/02_final_exoplanet_classification.ipynb` | Final executed notebook: preprocessing, tuning, 5-fold grouped CV, ResNet-MLP, FT-Transformer, disjoint calibration, ablations, failure analysis, locked test evaluation, and fresh artifact reload check |
| `models/final_xgboost_model.json` | Portable native JSON export of the trained final XGBoost classifier (`500` trees, `max_depth=6`, `lr=0.03`) |
| `models/final_xgboost_model.joblib` | Joblib binary artifact of the trained XGBoost classifier (`scikit-learn==1.6.1`, `xgboost==3.4.1`) |
| `models/final_resnet_mlp.pt` | Trained ResNet-MLP state dict (`68,993` params) and disjoint temperature $T^* = 1.074$ |
| `models/final_ft_transformer.pt` | Trained FT-Transformer state dict (`30,593` params) and disjoint temperature $T^* = 1.058$ |
| `models/final_neural_preprocessor_stats.json` | Version-independent JSON export of fitted median imputer statistics, missingness indicator indices, and Z-score scaler parameters |
| `models/final_neural_preprocessor.joblib` | Joblib binary artifact of the fitted preprocessing pipeline (`scikit-learn==1.6.1`) |
| `figures/fig1_roc_pr_curves.png` | Held-out test set ROC and Precision-Recall curves ($300$ DPI) |
| `figures/fig2_diagnostics_calibration.png` | Held-out test set Reliability Diagram and Complexity vs. Performance trade-off plot ($300$ DPI) |
| `Reports/CMPS_261 final report.pdf` | Final 7-section project report |
| `Reports/Report progress.pdf` | Milestone 1 progress report |
| `Poster/CMPS_261 poster.pdf` | Final 7-component project poster |
| `requirements.txt` | Pinned Python dependencies matching the training environment |

## Environment Setup & Portable Model Loading

To reproduce the training environment (Python 3.13 / CUDA 13.0):

```bash
pip install -r requirements.txt