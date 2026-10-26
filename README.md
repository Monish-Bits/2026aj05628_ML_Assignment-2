# Student Dropout & Academic Success: Multi-Model Classification

**ML Assignment 2 · M.Tech (AIML) · 2026aj05628**

- **Live Streamlit app:** <https://2026aj05628mlassignment-2.streamlit.app/>
- **GitHub repository:** <https://github.com/Monish-Bits/2026aj05628_ML_Assignment-2>

---

## a. Problem statement

Higher-education institutions lose a significant share of students before graduation. Identifying at-risk students early lets institutions step in with tutoring, financial aid or counselling while it can still help.

The task is to **predict a student's academic outcome**: **Dropout**, **Enrolled** (still studying after the normal duration) or **Graduate**. The prediction uses information available at enrolment (demographics, socio-economic background, application data) together with first- and second-semester academic performance. This is a **3-class, imbalanced classification problem**.

Five classifiers are trained on the same data and compared on six evaluation metrics. The models are served through an interactive Streamlit app that scores an uploaded labelled test file without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | *Predict Students' Dropout and Academic Success* |
| Source | UCI Machine Learning Repository, dataset id **697**: <https://archive.ics.uci.edu/dataset/697> |
| Origin | Polytechnic Institute of Portalegre (Portugal), students enrolled 2008/09 to 2018/19 across 17 undergraduate degrees |
| Instances | **4,424** (requirement: ≥ 500) |
| Features | **36** (requirement: ≥ 12): 29 integer, 7 real-valued |
| Target | `Target`: Graduate 2,209 (49.9 %), Dropout 1,421 (32.1 %), Enrolled 794 (17.9 %) |
| Missing values | None |

**Feature groups**

| Group | Features |
|---|---|
| Demographic | Marital Status, Nacionality, Gender, Age at enrollment, Displaced, International, Educational special needs |
| Application / programme | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Admission grade |
| Socio-economic | Mother's / Father's qualification, Mother's / Father's occupation, Debtor, Tuition fees up to date, Scholarship holder |
| Academic, 1st & 2nd semester | Curricular units credited / enrolled / evaluations / approved / grade / without evaluations (6 per semester) |
| Macro-economic | Unemployment rate, Inflation rate, GDP |

**Preprocessing** (inside each saved sklearn `Pipeline`, so the app takes raw columns):

- The 9 integer-coded **nominal** columns (Marital Status, Application mode, Course, Previous qualification, Nacionality, parents' qualification and occupation) are **one-hot encoded**. Codes seen fewer than 20 times are pooled into an "infrequent" bucket, and unseen codes are handled safely.
- The remaining 27 columns are **standardised** (z-score).
- **Split:** stratified 80 / 20 hold-out (`random_state=2026`), giving 3,539 training rows and 885 test rows. The 885 test rows are saved as [`test_data.csv`](test_data.csv) and never touch training or tuning.
- **Tuning:** `GridSearchCV` with 5-fold stratified CV on the training split, optimising **macro-F1** (chosen over accuracy because of the class imbalance).

## c. GitHub repository link

<https://github.com/Monish-Bits/2026aj05628_ML_Assignment-2>

```
├── .streamlit/config.toml     # app theme
├── app.py                     # Streamlit app (loads saved models, never retrains)
├── train_models.py            # data split, tuning, training, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv              # 885-row hold-out set (36 features + Target)
├── data/
│   └── students_dropout_academic_success.csv   # full raw UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── test_metrics.csv       # comparison table below, as CSV
    └── metadata.json          # feature list, classes, best hyper-parameters
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 885-row hold-out test set.

| Model | Best hyper-parameters (5-fold CV) |
|---|---|
| Logistic Regression | C = 0.1, class_weight = balanced |
| Decision Tree | criterion = gini, max_depth = None, min_samples_leaf = 20 |
| K-Nearest Neighbors | k = 5, Manhattan distance (p = 1), uniform weights |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 |
| Support Vector Machine | RBF kernel, C = 10, gamma = 0.01, class_weight = balanced; probabilities via Platt (sigmoid) calibration |

### Comparison table: hold-out test set (885 rows)

Precision, Recall and F1 are **macro-averaged** over the three classes. AUC is the **one-vs-rest macro ROC-AUC**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7571 | **0.8881** | 0.7262 | **0.7295** | 0.7203 | 0.6183 |
| Decision Tree | 0.7198 | 0.8316 | 0.6639 | 0.6480 | 0.6540 | 0.5366 |
| KNN | 0.7051 | 0.8170 | 0.6429 | 0.6118 | 0.6198 | 0.5077 |
| Naive Bayes | 0.7017 | 0.8185 | 0.6527 | 0.6258 | 0.6341 | 0.5067 |
| Support Vector Machine | **0.7740** | 0.8647 | **0.7339** | 0.7185 | **0.7236** | **0.6308** |

Per-class recall on the test set (Dropout / Enrolled / Graduate): LR 0.72 / 0.65 / 0.82, DT 0.70 / 0.40 / 0.85, KNN 0.64 / 0.30 / 0.89, NB 0.62 / 0.39 / 0.86, SVM 0.73 / 0.54 / 0.89.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and very close to the top: highest **AUC (0.888)** and highest macro **Recall (0.730)**. With `class_weight=balanced` it recovers **65 % of the hard "Enrolled" minority class**, far more than any other model, at the cost of some Graduate recall (0.82). The fairly strong regularisation (C = 0.1) suggests the signal is largely linear. Approved units and grades in semesters 1 and 2, plus tuition status, separate the classes almost monotonically. It is also the fastest and most interpretable model. |
| Decision Tree | Mid-table (F1 0.654, MCC 0.537). CV picked an unrestricted depth but `min_samples_leaf = 20`, i.e. it relies on leaf size to limit overfitting. Its axis-aligned splits capture the "approved units" thresholds well (Graduate recall 0.85), but it finds only 40 % of Enrolled students. Its piece-wise constant leaf probabilities rank students noticeably worse than LR or SVM (AUC 0.832 vs ≈ 0.87 to 0.89). |
| KNN | Weakest by F1 (0.620) and AUC (0.817). With 36 raw features expanded to 120 after one-hot encoding, distances get diluted by many weakly informative dimensions. Neighbourhoods are therefore dominated by the majority Graduate class: Graduate recall is 0.89, but Enrolled recall is only 0.30. A small k (5) with Manhattan distance did best, but this kind of local similarity does not suit this mixed, partly categorical data. |
| Naive Bayes | Lowest accuracy (0.702) and MCC (0.507). The independence assumption is clearly violated: the 1st- and 2nd-semester curricular unit features are highly correlated (enrolled ↔ evaluations ↔ approved), so the same evidence is counted several times and probabilities become over-confident. A large `var_smoothing` (0.1) was needed to stabilise the Gaussians. It is still a reasonable, extremely fast baseline (AUC 0.819). |
| Support Vector Machine | **Best overall** on Accuracy (0.774), Precision (0.734), F1 (0.724) and MCC (0.631). The RBF kernel with balanced class weights adds a little non-linear flexibility on top of the largely linear signal, which LR cannot model. It keeps the best Dropout and Graduate recall while finding 54 % of Enrolled students. Its AUC (0.865) is a bit lower than LR's because its probabilities come from post-hoc Platt calibration rather than being learned directly. It is the slowest model to tune. |
| **Overall winner for this dataset?** | **Support Vector Machine (RBF)**: highest MCC, F1, accuracy and precision, so it gives the most balanced correct predictions across all three classes. **Logistic Regression is a very close second.** It wins on AUC and macro recall, and it would be preferred when the goal is to *flag as many at-risk (Enrolled/Dropout) students as possible* or when interpretability is needed. For every model, **"Enrolled" is the hardest class**: these students sit between the two other outcomes and are the main source of errors. |

---

## Streamlit app features

The app follows a guided two-step flow (**Step 1 · Upload test data → Step 2 · Model → Results**) with a custom theme in `.streamlit/config.toml`.

- **CSV upload** of labelled test data. The sample `test_data.csv` and an empty header-only template can be downloaded from the same panel.
- **Model selection dropdown** for the five classifiers, with a one-line description and the tuned hyper-parameters.
- **Evaluation metrics**: six KPI cards (Accuracy, AUC, Precision, Recall, F1, MCC), computed live on the uploaded data. Each card has a tooltip explaining the metric and a delta against the average of all five models.
- **Model performance** tab: **confusion matrix** (counts and row-%) and **classification report**, with the hardest class called out.
- **Compare models** tab: metric table with the best value per metric highlighted, a winner callout, and a ranking chart you can sort by any metric.
- **Predictions** tab: per-row predicted class, confidence and class probabilities. You can filter to correct or wrong rows and by actual class, and download the result as CSV.
- **Test data** tab: outcome distribution and a preview of the uploaded file.
- **Input validation**: missing columns, unreadable files, missing values and unknown target labels are reported clearly (error message or "rows skipped" badge) instead of crashing the app.

The app only calls `predict` / `predict_proba` on the saved pipelines; **no model is retrained on the uploaded data**.

## How to run locally

```bash
pip install -r requirements.txt     # Python 3.11
python train_models.py              # optional: re-creates model/*.joblib, test_data.csv, metrics
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

New app → select this repository → branch `main` → main file `app.py` → *Advanced settings*: Python **3.11** → Deploy.
