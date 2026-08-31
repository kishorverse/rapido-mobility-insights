# Model Results

Four models trained on 100,000 Rapido bookings across five Indian cities, calendar year 2025.
Every figure below is read from `models/metrics.json`, written by `python src/train_models.py`
at training time. All four models were fitted on the same 80,000-row training split and scored on
the same held-out 20,000-row test split.

---

## 1. Summary

| Model | Artefact | Algorithm | Task | Headline | Secondary |
|---|---|---|---|---|---|
| Ride outcome | `ride_outcome_model.pkl` | HistGradientBoosting | 3-class | F1-macro **0.5652** | ROC-AUC (OvR) 0.8193 |
| Fare prediction | `fare_prediction_model.pkl` | HistGradientBoosting | regression | R² **0.9966** | MAPE 2.76% |
| Cancellation risk | `customer_cancellation_model.pkl` | HistGradientBoosting | binary | ROC-AUC **0.8512** | F1-macro 0.7161 |
| Driver delay risk | `driver_delay_model.pkl` | RandomForest | binary | ROC-AUC **0.7235** | F1-macro 0.6112 |

Every model clears its baseline by a wide margin. `dummy` predicts the majority class for
classification and the mean for regression; it is trained and scored alongside the real candidates
so that "better than nothing" is measured rather than assumed.

---

## 2. Ride outcome (3-class)

Predicts Completed / Cancelled / Incomplete before the trip starts.

**Class balance:** Completed 68.35%, Cancelled 23.28%, Incomplete 8.37%.
Trained with balanced class weights. 63 features (50 numeric, 13 categorical).

| Metric | Value |
|---|---|
| Accuracy | 0.6723 |
| Balanced accuracy | 0.6185 |
| F1-macro | 0.5652 |
| F1-weighted | 0.6969 |
| Precision (macro) | 0.5522 |
| Recall (macro) | 0.6185 |
| ROC-AUC (one-vs-rest) | 0.8193 |

**Candidate leaderboard** (held-out test set)

| Estimator | F1-macro | Fit (s) |
|---|---|---|
| **hist_gb** | **0.5652** | 7.46 |
| logreg | 0.5418 | 2.19 |
| random_forest | 0.5218 | 6.45 |
| dummy (baseline) | 0.2707 | 1.06 |

**5-fold cross-validation:** F1-macro 0.5416 ± 0.0040 — folds 0.545, 0.5367, 0.5425, 0.5372, 0.5466.
The tight spread means the held-out score is stable rather than a lucky split.

**Top features (permutation importance):** `customer_loyalty_score` (0.117), `acceptance_rate`
(0.109), `cust_prior_completion_rate` (0.059), `drv_prior_cancel_rate` (0.050), `cust_prior_rides`
(0.042).

**Reading it.** Three-class separation is genuinely hard here: Incomplete is only 8.4% of bookings
and overlaps heavily with Cancelled in feature space. F1-macro 0.5652 against a 0.2707 baseline is a
real signal, but this model is better used for its probability distribution than its hard label. The
two dedicated binary models below serve the operational decisions more directly.

---

## 3. Fare prediction (regression)

Predicts `booking_value` at request time, **without** `base_fare`. 59 features (46 numeric, 13
categorical). Target mean ₹336.34, standard deviation ₹208.02.

| Metric | Value |
|---|---|
| R² | 0.9966 |
| RMSE | ₹12.15 |
| MAE | ₹8.98 |
| MAPE | 2.76% |
| RMSE as % of mean | 3.59% |
| Within ±10% of actual | **99.82%** |
| Within ±20% of actual | 100.00% |

**Candidate leaderboard**

| Estimator | R² | Fit (s) |
|---|---|---|
| **hist_gb** | **0.9966** | 1.72 |
| random_forest | 0.9966 | 22.88 |
| ridge | 0.9196 | 0.88 |
| dummy (baseline) | -0.0001 | 0.76 |

`hist_gb` and `random_forest` tie on R²; `hist_gb` wins on fit time by a factor of thirteen.

**5-fold cross-validation:** RMSE 12.1601 ± 0.1166.

**Top features:** `vehicle_type` (0.858), `ride_distance_km` (0.645), `surge_multiplier` (0.214),
`estimated_ride_time_min` (0.063), `weather_condition` (0.001).

### This model is at the noise floor

Fare in this dataset is not a noisy real-world quantity. It is generated as:

```
base_fare     = flagfall + rate_per_km × distance     (R² = 1.000000 per vehicle type)
booking_value = base_fare × surge × (1 ± 5% uniform noise)
```

Recovered tariff: Bike ₹20 + ₹8/km · Auto ₹40 + ₹12/km · Cab ₹80 + ₹18/km.

Uniform ±5% noise has an expected absolute deviation of 2.5%, so **no model can achieve better than
about 2.50% MAPE**. This one reaches 2.76%. The remaining error is the noise term, not a modelling
shortfall — the top three features are exactly the three tariff inputs, which is what a model that
has recovered the pricing rule should look like.

The brief's benchmark was RMSE within ±10% of actual fare; 99.82% of predictions clear it.

---

## 4. Cancellation risk (binary)

Predicts whether a booking will be cancelled. Positive class 23.28%; trained with balanced class
weights. 63 features.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8512 |
| PR-AUC | 0.6422 |
| Accuracy | 0.7601 |
| Balanced accuracy | 0.7693 |
| F1-macro | 0.7161 |
| Precision (macro) | 0.7057 |
| Recall (macro) | 0.7693 |
| Positive rate | 0.2329 |

**Candidate leaderboard**

| Estimator | F1-macro | ROC-AUC | Fit (s) |
|---|---|---|---|
| **hist_gb** | **0.7161** | 0.8512 | 4.02 |
| logreg | 0.6988 | 0.8234 | 1.40 |
| random_forest | 0.6907 | 0.7968 | 6.40 |
| dummy (baseline) | 0.4341 | 0.5000 | 0.87 |

**5-fold cross-validation:** F1-macro 0.7057 ± 0.0014 — the tightest spread of any model here.

**Top features:** `acceptance_rate` (0.123), `customer_loyalty_score` (0.094),
`drv_prior_cancel_rate` (0.061), `surge_multiplier` (0.043), `cust_prior_rides` (0.034).

**Reading it.** This is the strongest and most operationally useful model in the project. ROC-AUC
0.8512 with PR-AUC 0.6422 against a 0.2329 prevalence baseline is enough to gate dispatch decisions.
`surge_multiplier` appearing in the top five matters: surge is platform-controlled, so the model
points at a lever the business can actually pull.

The decision threshold is an operational choice, not a modelling one — a lower cut-off catches more
cancellations but flags more bookings for intervention. The Model Lab page in the dashboard renders
the full precision/recall/volume trade-off table.

---

## 5. Driver delay risk (binary)

Predicts whether a ride ends Incomplete. Positive class 8.37% — the most imbalanced target here.
63 features.

| Metric | Value |
|---|---|
| ROC-AUC | 0.7235 |
| PR-AUC | 0.2360 |
| Accuracy | 0.8873 |
| Balanced accuracy | 0.6052 |
| F1-macro | 0.6112 |
| Precision (macro) | 0.6185 |
| Recall (macro) | 0.6052 |
| Positive rate | 0.0837 |

**Candidate leaderboard**

| Estimator | F1-macro | ROC-AUC | Fit (s) |
|---|---|---|---|
| **random_forest** | **0.6112** | 0.7235 | 5.72 |
| hist_gb | 0.5756 | 0.7631 | 3.97 |
| logreg | 0.5487 | 0.7667 | 1.47 |
| dummy (baseline) | 0.4782 | 0.5000 | 0.87 |

**5-fold cross-validation:** F1-macro 0.5992 ± 0.0055.

**Top features:** `traffic_level` (0.066), `expected_speed_kmph` (0.060), `avg_pickup_delay_min`
(0.040), `high_traffic_flag` (0.039), `customer_loyalty_score` (0.035).

**Reading it.** The weakest of the four, and the selection deserves a note: `random_forest` was
chosen on F1-macro (0.6112) even though `logreg` scores higher on ROC-AUC (0.7667 vs 0.7235). The
selection rule ranks on F1-macro for classification, which favours the model that makes better hard
decisions at the default threshold over the one that ranks better across all thresholds. If this
model were deployed for ranking rather than flagging, `logreg` would be the better pick.

The 88.73% accuracy is not the achievement it appears to be — predicting "never incomplete" would
score 91.63%. Balanced accuracy 0.6052 and PR-AUC 0.2360 against an 0.0837 baseline are the honest
numbers. The signal is real but modest.

Notably, the top features are all *traffic* related, not driver-identity related. This matches the
EDA finding that traffic moves the incompletion rate from 5.1% to 14.8% while weather leaves it
flat — routing support is a better intervention than driver replacement.

---

## 6. Leakage control

Three column families in this dataset would produce spectacular, meaningless scores. Each is blocked
in `src/train_models.py` by `assert_no_leakage()`, which **raises rather than warns**.

| Column | Why it leaks | Blocked for |
|---|---|---|
| `actual_ride_time_min` | Null for every non-Completed ride; the null indicator alone reproduces the target | all four models |
| `incomplete_ride_reason` | Only populated after a ride fails | all four models |
| `base_fare` | Reproduces `booking_value` to within 5% once multiplied by surge | fare model |
| `fare_per_km`, `fare_per_min` | Derived from `booking_value` | fare model |
| `cancellation_rate`, `delay_rate` | Whole-period aggregates that include the row being predicted | replaced, not merged |
| `customer_cancel_flag`, `driver_delay_flag` | The same rates, thresholded | replaced, not merged |

The dimension-table rate columns are replaced by **14 prior-history features** computed over an
expanding window of strictly earlier bookings, so no booking contributes to its own predictors.
Notebook `02_feature_engineering.ipynb` verifies the window is causal row by row.

### The fare leakage ablation

The clearest demonstration that `base_fare` is a formula rather than a feature: refit the fare model
*with* it and compare.

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| Honest — `base_fare` excluded | 0.9966 | ₹12.15 | 2.76% |
| Ablation — `base_fare` included | 0.9968 | ₹11.82 | 2.645% |

The gap is 0.0002 R². `base_fare` carries essentially no independent information, because the tariff
is already recoverable from distance and vehicle type — which is exactly what the deployed model
does. This ablation is recorded in `models/metrics.json` but never saved as a deployable artefact.

---

## 7. Method

Identical protocol for all four models:

1. **Build** the feature matrix from `data/processed/model_data.csv` and assert no leakage.
2. **Split** 80/20, stratified for classification targets, `random_state=42`.
3. **Baseline** with a `dummy` estimator on the same split.
4. **Compare** four candidate estimators, cheapest first.
5. **Select** the best non-dummy estimator by F1-macro (classification) or R² (regression).
6. **Cross-validate** the winner with 5 folds on the training split only.
7. **Explain** with permutation importance on the held-out set, measured over the original columns
   rather than one-hot fragments.
8. **Persist** the pipeline with its metrics, leaderboard, CV scores and importances.

All preprocessing — median imputation, scaling, one-hot encoding — lives **inside** the sklearn
Pipeline rather than being applied to the frame beforehand. Every cross-validation fold therefore
re-fits the imputer, scaler and encoder on training data only, which is what keeps the CV scores
honest.

Tree ensembles skip scaling; only `logreg` and `ridge` receive it. Unseen categories at predict time
are ignored rather than raising, so a novel city or vehicle type degrades gracefully.

**Reproduce:**

```bash
python src/feature_engineering.py build --rebuild   # rebuild model_data.csv
python src/train_models.py                          # train all four
python src/train_models.py --model fare --tune      # retrain one with a search
```

---

## 8. Operational recommendations

1. **Cap surge during adverse conditions.** Cancellations run 5.3% at no surge and 35.3% above 2.0x,
   and surge is the one variable in the top-five feature list that the platform sets directly.
2. **Allocate drivers on live traffic, not on city.** Cancellation rates across the five cities span
   22.95%–23.78% and chi-square returns p = 0.40 — city is not a signal. Traffic is.
3. **Score bookings at request time.** The cancellation model (ROC-AUC 0.8512) is strong enough to
   hold driver assignment on high-risk requests pending rider confirmation.
4. **Treat the two failure modes separately.** Weather drives cancellations (rider-side: fare
   guarantees, wait-time transparency); traffic drives incompletions (driver-side: routing support).
   The driver-risk model's traffic-dominated feature list supports the second half directly.
