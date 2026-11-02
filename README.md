# NASA Kepler Exoplanet Candidate Classification (CMPS 261)

**Course:** CMPS 261 Machine Learning
**Track:** Track A 4.1 Tabular (NASA Kepler Objects of Interest)  
**Group 4:** Joe Habre & Joy Habib Najem  

---

## Project Overview
This repository contains the end-to-end machine learning workflow for classifying **NASA Kepler Objects of Interest (KOIs)** into **Potential Exoplanets** (`CONFIRMED` $\cup$ `CANDIDATE`, $y=1$) versus **False Positives** (`FALSE POSITIVE`, $y=0$). Beyond aggregate predictive metrics, our experimental design investigates:
1. **Data Auditing & Leakage Prevention:** Identifying and removing post-vetting diagnostic flags (`koi_fpflag_*`), disposition scores (`koi_score`), and post-confirmation identifiers (`kepler_name`) that artificially inflate accuracy.
2. **Evaluation Design & Split Sensitivity:** Comparing **Host-Star Grouped Splitting** (`GroupShuffleSplit` / `GroupKFold` on `kepid`) against standard random splitting to prevent multi-candidate stellar host leakage.
3. **The Model Ladder:**
   - **Level 1 (Classical Baselines):** Regularized Logistic Regression ($L_2$), Decision Trees (Information Gain), Random Forests (Bootstrap Aggregation + Feature Subsampling $K=\sqrt{N}$), and Tuned XGBoost (Second-order gradients/Hessians, $L_2$ leaf regularization $\lambda$, complexity penalty $\gamma$, and native missing-value routing).
   - **Level 2 (Sophisticated Models):** Deep Tabular **ResNet-MLP** (skip connections $y = F(x) + x$, BatchNorm, ReLU, Dropout, AdamW, Early Stopping) and **Tabular Self-Attention (FT-Transformer)** with post-hoc **Temperature Scaling** ($\sigma(z/T)$) for confidence calibration.
4. **Diagnostics, Subgroup Fairness & Failure Analysis:** Learning curves ($J_{\text{train}}$ vs. $J_{\text{cv}}$), reliability diagrams (Expected Calibration Error), and manual error categorization across signal-to-noise (`koi_model_snr`), grazing impact parameters (`koi_impact`), and evolved host-star subgroups (`koi_slogg`).
