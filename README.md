# NASA Kepler Exoplanet Candidate Classification

**Course:** CMPS 261 Machine Learning

**Track:** Track A, 4.1 Tabular (NASA Kepler Objects of Interest)

**Group 4:** Joe Habre and Joy Habib Najem

We classify Kepler Objects of Interest (KOIs) as potential exoplanets or false positives. The positive class combines `CONFIRMED` and `CANDIDATE` objects (`y = 1`); `FALSE POSITIVE` objects form the negative class (`y = 0`). Candidate labels are provisional, so this task predicts archive dispositions rather than establishing that an object is a real planet.

The repository contains our progress-report experiments. We compare classical classifiers with a residual multilayer perceptron (ResNet-MLP), while examining feature leakage, validation design, calibration, and model failures.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/cumulative.csv` | Committed NASA KOI dataset snapshot |
| `notebooks/01_milestone_baseline.ipynb` | Preprocessing, model training, validation, and plots |
| `figures/` | ROC/precision-recall curves and a reliability diagram |
| `Reports/CMPS_261.pdf` | Draft progress report |
| `Project Proposal.docx` | Initial topic proposal |
| `requirements.txt` | Python dependencies with minimum versions |

## Data and preprocessing

The committed CSV contains 9,564 objects, 153 columns, and 8,214 distinct host stars. Its dispositions are 2,748 confirmed objects, 1,977 candidates, and 4,839 false positives. Column definitions are available in the [NASA Exoplanet Archive documentation](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html). The notebook includes the NASA TAP query used to download the cumulative table.

We exclude the disposition columns, `koi_score`, the four `koi_fpflag_*` fields, and identifier/provenance fields from the model inputs. The notebook selects numerical columns with less than 50% missingness in the development partition, then adds a transit depth/radius feature, a transit duty-cycle feature, and six logarithmic transformations.

Logistic regression, random forest, and ResNet-MLP use median imputation with missingness indicators fitted within each training fold. Logistic regression and ResNet-MLP also use standardization. XGBoost receives missing values directly. Some retained columns are vetting statistics; their availability at prediction time still needs to be justified before describing the feature set as entirely pre-vetting.

## Current experiments

- **Validation:** `GroupShuffleSplit` with seed 42 reserves 20% of host stars for a final test partition. The resulting partitions contain 7,662 development rows and 1,902 test rows, with no shared stars. The test partition is not evaluated by the current notebook.
- **Logistic regression:** A three-fold `GroupKFold` search compares `C` values of 0.01, 0.1, 1.0, and 10.0 using mean ROC-AUC. The selected configuration is then included in the five-fold development evaluation.
- **Random forest and XGBoost:** Both are evaluated with fixed parameter settings. The notebook does not implement an XGBoost hyperparameter search, despite its current display label saying "Tuned Classical Baseline."
- **ResNet-MLP:** A residual network with batch normalization and dropout is trained with AdamW for 50 epochs per fold. The code retains predictions from the epoch with the lowest validation loss; it does not stop training early. Model and tensor operations currently run on the CPU.
- **Outputs:** Five-fold grouped validation produces mean and standard deviation for ROC-AUC, average precision (currently labelled PR-AUC), and F1, plus mean precision, recall, ECE, and fit time. Plot legends use pooled out-of-fold metrics, which can differ from fold averages.
- **Failure analysis:** XGBoost error rates are calculated for low-SNR objects, high-SNR objects, and objects with missing stellar radius, together with subgroup sizes.