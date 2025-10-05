# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

# Model Details

Task: Binary classification — predict whether an individual’s income is >50K vs <=50K.

Algorithm: RandomForestClassifier (n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1).

Preprocessing: ml/data.py::process_data

OneHotEncoder for categorical features

LabelBinarizer for salary (>50K → 1, <=50K → 0)

# Features:

Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

Numeric: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

Artifacts: model/model.pkl, model/encoder.pkl, model/lb.pkl; slice results in slice_output.txt.

Intended Use

Primary: Educational demo of an end-to-end ML pipeline (training, slicing, and FastAPI serving).

Not for: Any real decisions affecting access to employment, credit, housing, healthcare, or similar high-stakes outcomes.

# Training Data

Source: data/census.csv (Census/Adult Income–style dataset).

Label: salary with classes >50K, <=50K.

Notes: Class imbalance is present (positive class is minority).

# Evaluation Data

Split: 80/20 train–test with stratification on salary (random_state=42).

Transform: Test data encoded using train-fit encoder/label binarizer.

Slices: Per-value metrics for each categorical feature; results written to slice_output.txt.

# Metrics

Overall test metrics (from your latest run):

Precision: 0.7433

Recall: 0.6205

F1: 0.6764
(Computed via compute_model_metrics with β=1, zero_division=1.)

# Slice Metrics (highlights)

Perfect scores on very tiny groups (Count < 10–20) are unreliable; focus on slices with Count ≥ 100 for stable signals.

workclass (≥100):

Best: Self-emp-inc — F1 0.7769 (N=222)

Also large: Private — F1 0.6725 (N=4,597)

Lowest: ? (unknown) — F1 0.5625 (N=363)

education (≥100):

Best: Prof-school — F1 0.8736 (N=114); Masters — F1 0.8442 (N=318); Bachelors — F1 0.7540 (N=1,096)

Lowest: 7th-8th — F1 0.0000 (N=120)

Large/low: HS-grad — F1 0.4779 (N=2,120); Some-college — F1 0.5726 (N=1,475)

marital-status (≥100):

Best: Married-civ-spouse — F1 0.6998 (N=2,977)

Lowest: Divorced — F1 0.4490 (N=883)

Large/low: Never-married — F1 0.5455 (N=2,181)

occupation (≥100):

Best: Prof-specialty — F1 0.7956 (N=818); Exec-managerial — F1 0.7830 (N=808)

Lowest: Other-service — F1 0.2667 (N=684)

Other low: Machine-op-inspct — F1 0.3944 (N=392); Transport-moving — F1 0.4167 (N=322); Adm-clerical — F1 0.5072 (N=819)

relationship (≥100):

Best: Wife — F1 0.7166 (N=336); Husband — F1 0.6985 (N=2,607)

Lowest: Own-child — F1 0.3333 (N=1,032); Unmarried — F1 0.4828 (N=712); Not-in-family — F1 0.5021 (N=1,636)

race (≥100):

Best: White — F1 0.6798 (N=5,533)

Lowest: Asian-Pac-Islander — F1 0.6263 (N=200)

Black: F1 0.6582 (N=662)

sex (≥100):

Female — F1 0.6604 (N=2,158)

Male — F1 0.6792 (N=4,355)

native-country (≥100):

? (unknown) — F1 0.7500 (N=125)

United-States — F1 0.6773 (N=5,835)

Lowest large: Mexico — F1 0.5000 (N=111)

Small-support perfects (e.g., “Never-worked”, “Without-pay”, some countries) likely reflect overfitting/variance—treat cautiously.

# Ethical Considerations

Sensitive features: race, sex are present; proxies (e.g., relationship, marital-status) may encode correlated biases.

Disparate performance: Several sizable groups show substantially lower F1 (e.g., HS-grad, Other-service, Own-child). Assess downstream fairness metrics (e.g., equal opportunity) before any deployment.

Misuse: This educational model must not be used for real hiring, lending, housing, or similar decisions.

# Caveats and Recommendations

Data drift & representativeness: Retrain and re-evaluate regularly; monitor overall and slice metrics.

Thresholding: If you move to probability outputs, tune thresholds per slice to balance precision/recall.

# Remediation ideas:

Reweighting or class-balanced sampling for low-F1, high-support slices (e.g., Other-service, HS-grad, Own-child).

Add interaction features; try calibrated linear models or gradient boosting; run simple fairness-aware post-processing (e.g., threshold adjustments per group).

Address ?/unknown categories explicitly (quality checks, consistent missing-value handling).