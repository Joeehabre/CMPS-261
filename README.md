# NASA Kepler Exoplanet Candidate Classification (CMPS 261)

**Course:** CMPS 261 — Introduction to Artificial Intelligence / Machine Learning (American University of Beirut)  
**Track:** Track A — 4.1 Tabular (NASA Kepler Objects of Interest)  
**Group 4:** Joe Habre & Joy Habib Najem  

---

## 1. Project Overview
This repository contains the end-to-end machine learning workflow for classifying **NASA Kepler Objects of Interest (KOIs)** into **Potential Exoplanets** (`CONFIRMED` $\cup$ `CANDIDATE`, $y=1$) versus **False Positives** (`FALSE POSITIVE`, $y=0$). Beyond aggregate predictive metrics, our experimental design investigates:
1. **Data Auditing & Leakage Prevention:** Identifying and removing post-vetting diagnostic flags (`koi_fpflag_*`), disposition scores (`koi_score`), and post-confirmation identifiers (`kepler_name`) that artificially inflate accuracy.
2. **Evaluation Design & Split Sensitivity:** Comparing **Host-Star Grouped Splitting** (`GroupShuffleSplit` / `GroupKFold` on `kepid`) against standard random splitting to prevent multi-candidate stellar host leakage.
3. **The Model Ladder:**
   - **Level 1 (Classical Baselines):** Regularized Logistic Regression ($L_2$), Decision Trees (Information Gain), Random Forests (Bootstrap Aggregation + Feature Subsampling $K=\sqrt{N}$), and Tuned XGBoost (Second-order gradients/Hessians, $L_2$ leaf regularization $\lambda$, complexity penalty $\gamma$, and native missing-value routing).
   - **Level 2 (Sophisticated Models):** Deep Tabular **ResNet-MLP** (skip connections $y = F(x) + x$, BatchNorm, ReLU, Dropout, AdamW, Early Stopping) and **Tabular Self-Attention (FT-Transformer)** with post-hoc **Temperature Scaling** ($\sigma(z/T)$) for confidence calibration.
4. **Diagnostics, Subgroup Fairness & Failure Analysis:** Learning curves ($J_{\text{train}}$ vs. $J_{\text{cv}}$), reliability diagrams (Expected Calibration Error), and manual error categorization across signal-to-noise (`koi_model_snr`), grazing impact parameters (`koi_impact`), and evolved host-star subgroups (`koi_slogg`).

---

## 2. Repository Structure
```text
├── data/
│   └── cumulative.csv                  # Official NASA Exoplanet Archive KOI cumulative dataset (auto-downloaded)
├── notebooks/
│   └── 01_milestone_baseline.ipynb     # Milestone 1: Auditing, Preprocessing, Model Ladder, Calibration & Diagnostics
├── figures/
│   ├── fig1_roc_pr_curves.png          # 5-Fold Grouped CV ROC-AUC and PR-AUC curves
│   ├── fig2_learning_curves.png        # Bias-Variance diagnosis: J_train vs J_cv learning curves
│   ├── fig3_calibration_curve.png      # Reliability diagram & Temperature Scaling (T*)
│   └── fig4_failure_subgroups.png      # Stratified failure rate across SNR, Impact Parameter, and Stellar Type
├── report/
│   └── Group4_Progress_Report.tex      # 2-Page Progress Report LaTeX source
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md
