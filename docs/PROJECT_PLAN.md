# Rapido — Intelligent Mobility Insights
## Build Plan & Function Inventory

**Project:** Ride Patterns, Cancellations & Fare Forecasting
**Stack:** Python · pandas · scikit-learn · MySQL · Streamlit · Plotly
**Timeline:** 10 days · **Evaluation:** 60 marks (code quality, docs, modularity, presentation, task accomplishment, mock Q&A)

---

## 1. Data Profile (verified, not assumed)

| File | Rows | Grain | Role |
|---|---|---|---|
| `bookings.csv` | 100,000 | one booking | fact table, all 4 model targets |
| `customers.csv` | 10,000 | one customer | customer dimension + history |
| `drivers.csv` | 5,000 | one driver | driver dimension + history |
| `location_demand.csv` | 17,941 | city × location × hour × vehicle | demand aggregate |
| `time_features.csv` | 8,760 | one hour of 2025 | calendar dimension |

Date range 2025-01-01 → 2025-12-31. No duplicate `booking_id`. **No orphan FKs** — every `customer_id` and `driver_id` in bookings exists in its dimension. Cities 5, vehicle types 3, traffic levels 3, weather conditions 3.

**Target distribution (`booking_status`):** Completed 68,346 (68.3%) · Cancelled 23,284 (23.3%) · Incomplete 8,370 (8.4%) — moderate imbalance, minority class ~8%.

### 1.1 Missingness is not random — it is the label

| Column | Nulls | Cause |
|---|---|---|
| `actual_ride_time_min` | 31,654 | **null for exactly every non-Completed ride** |
| `incomplete_ride_reason` | 91,630 | populated only for the 8,370 Incomplete rides |

Both are *post-outcome* columns. Do **not** impute them and do **not** feed them to the outcome model — `actual_ride_time_min.isna()` alone predicts the target with 100% accuracy. They are EDA/diagnostic columns only.

### 1.2 Three leakage traps to design around

1. **Outcome leak** — `actual_ride_time_min`, `incomplete_ride_reason` (above).
2. **Fare is near-deterministic** — `booking_value / (base_fare × surge_multiplier)` has mean 0.99993, sd 0.029, range [0.95, 1.05]. Fare is `base_fare × surge × ±5% noise`. A regressor given `base_fare` scores R² ≈ 0.999 and proves nothing. **Ship two fare models:** (a) *pre-quote* model excluding `base_fare` — predicts from distance, estimated time, vehicle, city, traffic, weather, hour, surge; this is the real business model; (b) the trivial baseline, reported as an EDA finding that documents the pricing formula. Say this out loud in the evaluation — it reads as rigour, not as a gap.
3. **Dimension flags are thresholded targets** — `customer_cancel_flag` = `cancellation_rate > 0.20`; `driver_delay_flag` = `delay_rate > 0.10`. Verified exactly. Predicting the flag from the rate is circular. Define the risk targets **at booking level** instead (§4.3, §4.4) and use the customer/driver history columns as *features*, computed leave-one-out so the history never includes the row being predicted.

### 1.3 Class balance of secondary targets
`customer_cancel_flag` 53/47 · `driver_delay_flag` 87/13 (needs `class_weight='balanced'`).

### 1.4 Scope items the data cannot support

| Requested | Reality | Substitution |
|---|---|---|
| Payment method usage patterns (EDA) | **No payment column exists** in any of the 5 files | Vehicle-type × surge usage-pattern analysis; deviation noted in README |
| `City_Pair = Pickup City + Drop City` | Every booking is **intra-city**; `Loc_1..Loc_50` are reused identically across all 5 cities, so bare codes collide | `city_route_pair = city + pickup + drop`; `locations` dimension keyed on (city, code) |
| `is_holiday` feature | **0 for all 8,760 hours** of 2025 — zero variance | Dropped via `config.ZERO_VARIANCE_COLUMNS` |
| `demand_level` three-way split | Only `Low` and `Medium` present — no `High` | Treated as binary |

Also note: 1,979 bookings have `pickup_location == drop_location`. Keep them (they are real same-zone trips) but flag with `is_same_zone`.

---

## 2. Architecture

Mirrors the Project_2 layout you already use, so the reviewer sees a consistent house style.

```
Project_3/
├── app.py                        # Streamlit entry, router, global filters
├── config.py                     # paths, DB creds, constants, model params
├── requirements.txt
├── README.md                     # setup, objectives, demo walkthrough
├── rapido/                       # importable package — all logic lives here
│   ├── __init__.py
│   ├── io.py                     # raw load + cached reads
│   ├── cleaning.py               # per-file cleaners
│   ├── features.py               # feature engineering
│   ├── schema.py                 # DDL, indexes
│   ├── db.py                     # connection, execute, read_sql
│   ├── etl.py                    # clean -> load -> verify pipeline
│   ├── queries.py                # named parameterised SQL for dashboard
│   ├── charts.py                 # Plotly figure builders
│   ├── stats.py                  # chi-square / ANOVA significance tests
│   └── models/
│       ├── __init__.py
│       ├── dataset.py            # model matrices, splits, leakage guard
│       ├── pipeline.py           # sklearn ColumnTransformer + estimator
│       ├── train.py              # 4 training entry points + tuning
│       ├── evaluate.py           # metric blocks, confusion matrix, curves
│       ├── explain.py            # feature importance / permutation
│       └── registry.py           # save/load/version artefacts
├── app_pages/
│   ├── _helpers.py  overview.py  demand.py  cancellations.py
│   ├── fares.py  customers.py  drivers.py  model_lab.py  predict.py
├── scripts/
│   ├── profile_raw.py  run_etl.py  train_all.py  make_insights.py
├── notebooks/  01_eda.ipynb
├── models/                       # .joblib artefacts + metrics.json (gitignored)
├── data/  raw/  processed/       # parquet cache
├── tests/   test_cleaning.py  test_features.py  test_dataset.py  test_queries.py
└── docs/  PROJECT_PLAN.md  INSIGHTS.md  data_quality_report.md
```

**Rule:** no business logic inside `app_pages/`. Pages call `rapido.*` and render. This is the *Code Reusability (10 marks)* line item.

---

## 3. Database Design (3NF)

```
cities(city_id PK, city_name)
locations(location_id PK, city_id FK, location_code)
vehicle_types(vehicle_type_id PK, vehicle_name)

customers(customer_id PK, gender, age, city_id FK, signup_days_ago,
          preferred_vehicle_type_id FK, total_bookings, completed_rides,
          cancelled_rides, incomplete_rides, cancellation_rate,
          avg_customer_rating, customer_cancel_flag)

drivers(driver_id PK, age, city_id FK, vehicle_type_id FK, experience_years,
        total_assigned_rides, accepted_rides, incomplete_rides, delay_count,
        acceptance_rate, delay_rate, avg_driver_rating,
        avg_pickup_delay_min, driver_delay_flag)

time_features(datetime PK, hour_of_day, day_of_week, is_weekend,
              is_holiday, peak_time_flag, season)

location_demand(demand_id PK, city_id FK, location_id FK, hour_of_day,
                vehicle_type_id FK, total_requests, completed_rides,
                cancelled_rides, avg_wait_time_min, avg_surge_multiplier,
                demand_level)

bookings(booking_id PK, booking_ts, city_id FK, pickup_location_id FK,
         drop_location_id FK, vehicle_type_id FK, customer_id FK, driver_id FK,
         ride_distance_km, estimated_ride_time_min, actual_ride_time_min NULL,
         traffic_level, weather_condition, base_fare, surge_multiplier,
         booking_value, booking_status, incomplete_ride_reason NULL)
```

**Indexes:** `bookings(booking_ts)`, `bookings(city_id, booking_status)`, `bookings(customer_id)`, `bookings(driver_id)`, `bookings(vehicle_type_id, hour_of_day)`, `location_demand(city_id, hour_of_day)`. Justify each in the viva by the query it serves.

---

## 4. The Four Models

| # | Model | Type | Target | Primary metric |
|---|---|---|---|---|
| 1 | Ride Outcome | 3-class | `booking_status` | macro-F1 (+ per-class recall) |
| 2 | Fare (pre-quote) | regression | `booking_value` | RMSE, MAE, R², MAPE |
| 3 | Customer Cancellation Risk | binary | `status == Cancelled` | ROC-AUC + PR-AUC |
| 4 | Driver Delay Risk | binary | delay / incomplete outcome | recall @ fixed precision |

### 4.1 Feature set (pre-booking signal only)
`ride_distance_km`, `estimated_ride_time_min`, `surge_multiplier`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_holiday`, `peak_time_flag`, `season`, `city`, `pickup_location`, `drop_location`, `vehicle_type`, `traffic_level`, `weather_condition`, customer history block, driver history block, location-demand block.

### 4.2 Engineered features (spec Step 3 + additions)
`fare_per_km`, `fare_per_min`, `rush_hour_flag`, `long_distance_flag`, `city_pair`, `driver_reliability_score`, `customer_loyalty_score`, plus `expected_speed_kmph`, `surge_bucket`, `is_night_ride`, `demand_supply_ratio`, `adverse_conditions_flag`, `customer_tenure_bucket`.

> `fare_per_km` / `fare_per_min` derive from `booking_value` — they are **EDA and dashboard features only**, excluded from the fare model's X. `dataset.py` enforces this with an explicit blocklist, not by convention.

### 4.3 Customer risk target
`y = 1 if booking_status == 'Cancelled'`; features = customer history computed **leave-one-out** (subtract the current booking from the customer's aggregates) + trip context. Report a calibrated probability so ops can pick their own threshold.

### 4.4 Driver risk target
`y = 1 if booking_status == 'Incomplete'` (optionally `OR incomplete_ride_reason == 'Driver Delay'`). Features = driver history (leave-one-out) + traffic/weather exposure + acceptance behaviour.

### 4.5 Modelling protocol
Stratified 80/20 split → baseline (`DummyClassifier` / mean-predictor) → LogisticRegression / Ridge → RandomForest → HistGradientBoosting → tune the winner with `RandomizedSearchCV` then a narrow `GridSearchCV`. 5-fold stratified CV. For models 1 and 4, run **both** `class_weight='balanced'` and SMOTE and report which won — don't just apply one.

**Benchmark honesty:** the spec targets 85–90% accuracy. If the pre-quote fare model lands at R² ≈ 0.7 and outcome accuracy near 0.70, that is the real ceiling of pre-trip signal in this data. Present it with the leakage evidence rather than hitting 99% on leaked columns, and include the leaked run as an ablation to show you know the difference.

---

## 5. Complete Function Inventory

### `config.py`
```python
BASE_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, DB_CONFIG
RANDOM_STATE = 42
CITIES, VEHICLE_TYPES, TRAFFIC_LEVELS, WEATHER_CONDITIONS
RUSH_HOURS, LONG_DISTANCE_KM, LEAKY_COLUMNS

get_db_config() -> dict                      # env vars with .env fallback
get_model_path(name: str) -> Path
```

### `rapido/io.py`
```python
load_bookings(path=None) -> pd.DataFrame
load_customers(path=None) -> pd.DataFrame
load_drivers(path=None) -> pd.DataFrame
load_location_demand(path=None) -> pd.DataFrame
load_time_features(path=None) -> pd.DataFrame
load_all_raw() -> dict[str, pd.DataFrame]
save_processed(df, name) -> Path             # parquet
load_processed(name) -> pd.DataFrame
profile_dataframe(df, name) -> pd.DataFrame  # dtype / nulls / uniques / range
```

### `rapido/cleaning.py`
```python
standardise_columns(df) -> df                # snake_case, strip
parse_booking_datetime(df) -> df             # date + time -> booking_ts
coerce_numeric(df, cols) -> df
coerce_categorical(df, cols) -> df
handle_missing_bookings(df) -> df            # keeps outcome nulls as NaN, flags them
handle_missing_customers(df) -> df
handle_missing_drivers(df) -> df
detect_outliers_iqr(df, col) -> pd.Series    # boolean mask
cap_outliers(df, cols, method='iqr') -> df
drop_duplicates_by_key(df, key) -> df
validate_referential_integrity(bookings, customers, drivers) -> dict
validate_value_ranges(df, rules: dict) -> pd.DataFrame
clean_bookings(df) -> df                     # orchestrator
clean_customers(df) -> df
clean_drivers(df) -> df
clean_location_demand(df) -> df
clean_time_features(df) -> df
clean_all(raw: dict) -> dict
build_data_quality_report(before, after) -> str   # -> docs/data_quality_report.md
```

### `rapido/features.py`
```python
add_time_parts(df) -> df                     # hour, weekday, month, is_weekend
add_rush_hour_flag(df) -> df
add_night_ride_flag(df) -> df
add_long_distance_flag(df, threshold) -> df
add_city_route_pair(df) -> df              # city + pickup + drop
add_same_zone_flag(df) -> df
add_fare_ratios(df) -> df                    # fare_per_km, fare_per_min  [EDA only]
add_expected_speed(df) -> df
add_surge_bucket(df) -> df
add_adverse_conditions_flag(df) -> df        # heavy rain + high traffic
compute_driver_reliability_score(drivers) -> pd.Series
compute_customer_loyalty_score(customers) -> pd.Series
add_customer_tenure_bucket(df) -> df
merge_customer_features(bookings, customers) -> df
merge_driver_features(bookings, drivers) -> df
merge_time_features(bookings, time_features) -> df
merge_demand_features(bookings, location_demand) -> df
add_demand_supply_ratio(df) -> df
leave_one_out_customer_history(bookings) -> df    # anti-leakage aggregates
leave_one_out_driver_history(bookings) -> df
encode_categoricals(df, cols, method='onehot') -> (df, encoder)
build_feature_table(clean: dict) -> pd.DataFrame  # master orchestrator
```

### `rapido/schema.py`
```python
CREATE_TABLE_STATEMENTS: list[str]
INDEX_STATEMENTS: list[str]
DROP_STATEMENTS: list[str]
TABLE_ORDER: list[str]                       # FK-safe insert order

get_create_statements() -> list[str]
get_column_map(table) -> dict
```

### `rapido/db.py`
```python
get_connection()                             # context manager
execute(sql, params=None) -> int
executemany(sql, rows) -> int
read_sql(sql, params=None) -> pd.DataFrame
table_exists(name) -> bool
row_count(table) -> int
create_database() -> None
create_tables() -> None
create_indexes() -> None
drop_all_tables() -> None
bulk_insert(table, df, chunk_size=5000) -> int
truncate_table(table) -> None
healthcheck() -> dict
```

### `rapido/etl.py`
```python
extract() -> dict                            # raw frames
transform(raw) -> dict                       # clean + normalise to table shapes
build_dimension_tables(clean) -> dict        # cities, locations, vehicle_types
load(tables) -> dict                         # row counts per table
verify_load(expected: dict) -> pd.DataFrame
run_etl(rebuild=False) -> dict               # full pipeline with logging
```

### `rapido/queries.py`
One function per dashboard question — all parameterised, all returning DataFrames.
```python
q_kpi_summary(filters) -> df                 # bookings, completion %, revenue, avg fare
q_rides_by_hour(filters) -> df
q_rides_by_weekday(filters) -> df
q_rides_by_city(filters) -> df
q_monthly_trend(filters) -> df
q_cancellation_rate_by_city_hour(filters) -> df    # heatmap source
q_cancellation_by_vehicle(filters) -> df
q_cancellation_reasons(filters) -> df
q_status_split_by_traffic_weather(filters) -> df
q_distance_vs_fare(filters, sample=5000) -> df
q_fare_by_vehicle_city(filters) -> df
q_surge_by_hour(filters) -> df
q_revenue_by_city_vehicle(filters) -> df
q_rating_distribution(filters) -> df
q_top_pickup_locations(filters, limit=20) -> df
q_busiest_city_pairs(filters, limit=20) -> df
q_high_risk_customers(limit=50) -> df
q_unreliable_drivers(limit=50) -> df
q_customer_vs_driver_cancellations(filters) -> df
q_demand_level_distribution(filters) -> df
q_wait_time_by_demand(filters) -> df
q_peak_cancellation_windows(filters) -> df
build_where_clause(filters) -> (str, list)   # shared filter builder
```
> Target **20+ named queries** — it feeds the *Task Accomplishment* mark directly.

### `rapido/charts.py`
```python
kpi_row(metrics: dict) -> None
line_rides_by_hour(df) -> go.Figure
bar_rides_by_city(df) -> go.Figure
heatmap_cancellation(df) -> go.Figure
stacked_status_by_category(df, category) -> go.Figure
scatter_distance_fare(df) -> go.Figure
box_fare_by_vehicle(df) -> go.Figure
line_surge_by_hour(df) -> go.Figure
histogram_ratings(df) -> go.Figure
treemap_revenue(df) -> go.Figure
bar_top_locations(df) -> go.Figure
confusion_matrix_fig(cm, labels) -> go.Figure
roc_curve_fig(fpr, tpr, auc) -> go.Figure
pr_curve_fig(precision, recall, ap) -> go.Figure
feature_importance_fig(df, top_n=20) -> go.Figure
residual_plot_fig(y_true, y_pred) -> go.Figure
apply_theme(fig) -> go.Figure                # one place for colours / fonts
```

### `rapido/stats.py`
```python
chi_square_independence(df, col_a, col_b) -> dict     # traffic vs cancellation
anova_fare_by_group(df, group_col) -> dict            # fare across vehicle types
correlation_matrix(df, cols, method='pearson') -> df
ttest_two_groups(df, value_col, group_col) -> dict    # weekend vs weekday fare
cramers_v(df, col_a, col_b) -> float
summarise_tests(results: list) -> pd.DataFrame
```

### `rapido/models/dataset.py`
```python
LEAKY_BY_TARGET: dict[str, list[str]]        # explicit blocklists
assert_no_leakage(X, target) -> None         # raises on any blocked column
build_outcome_dataset(df) -> (X, y)
build_fare_dataset(df, include_base_fare=False) -> (X, y)
build_customer_risk_dataset(df) -> (X, y)
build_driver_risk_dataset(df) -> (X, y)
split_train_test(X, y, stratify=True) -> (X_tr, X_te, y_tr, y_te)
get_feature_types(X) -> (numeric_cols, categorical_cols)
resample_balanced(X, y, method='smote') -> (X, y)
```

### `rapido/models/pipeline.py`
```python
build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer
build_classifier(name, **params) -> Pipeline    # logreg | rf | hgb | dummy
build_regressor(name, **params) -> Pipeline     # ridge | rf | hgb | mean
get_param_grid(name, task) -> dict
```

### `rapido/models/train.py`
```python
train_baseline(X, y, task) -> dict
train_and_compare(X, y, task, candidates) -> pd.DataFrame   # leaderboard
tune_model(pipeline, param_grid, X, y, task, n_iter=30) -> (best, cv_results)
train_outcome_model(df) -> dict
train_fare_model(df) -> dict
train_customer_risk_model(df) -> dict
train_driver_risk_model(df) -> dict
train_all_models(df) -> dict                 # -> models/ + metrics.json
```

### `rapido/models/evaluate.py`
```python
classification_metrics(y_true, y_pred, y_proba=None) -> dict
regression_metrics(y_true, y_pred) -> dict   # RMSE, MAE, R2, MAPE, +/-10% hit-rate
confusion_matrix_df(y_true, y_pred, labels) -> df
roc_data(y_true, y_proba) -> (fpr, tpr, auc)
pr_data(y_true, y_proba) -> (precision, recall, ap)
cross_validate_model(pipeline, X, y, task, cv=5) -> df
compare_models(results: dict) -> pd.DataFrame
within_tolerance_rate(y_true, y_pred, tol=0.10) -> float   # the spec's +/-10% benchmark
```

### `rapido/models/explain.py`
```python
get_feature_names(pipeline) -> list[str]
tree_feature_importance(pipeline) -> pd.DataFrame
permutation_feature_importance(pipeline, X, y, task) -> pd.DataFrame
top_drivers_for_prediction(pipeline, row) -> pd.DataFrame   # single-row reasons
```

### `rapido/models/registry.py`
```python
save_model(pipeline, name, metrics, metadata) -> Path
load_model(name) -> (pipeline, metadata)
list_models() -> pd.DataFrame
save_metrics(name, metrics) -> Path
load_metrics(name=None) -> dict
model_exists(name) -> bool
```

### `app.py` and `app_pages/`
```python
# app.py
configure_page() -> None
sidebar_filters() -> dict                    # city, date range, vehicle, traffic, weather
render_navigation() -> str
main() -> None

# app_pages/_helpers.py
cached_query(name, filters) -> df            # @st.cache_data
cached_model(name)                           # @st.cache_resource
format_currency(x) -> str
format_pct(x) -> str
show_metric_delta(label, value, delta) -> None
paginate(df, page_size=50) -> df             # spec: no full-data loads
download_button(df, filename) -> None
empty_state(msg) -> None

# app_pages/overview.py       render()  - KPIs, volume trends, city split
# app_pages/demand.py         render()  - hour/weekday/location heatmaps, surge
# app_pages/cancellations.py  render()  - heatmap, reasons, traffic/weather driver
# app_pages/fares.py          render()  - distance-fare, per-km, revenue mix
# app_pages/customers.py      render()  - segments, loyalty, high-risk table
# app_pages/drivers.py        render()  - reliability leaderboard, delay analysis
# app_pages/model_lab.py      render()  - leaderboard, CM, ROC/PR, importance
# app_pages/predict.py        render()  - 4 live prediction forms
predict_outcome_form() -> None
predict_fare_form() -> None
predict_customer_risk_form() -> None
predict_driver_risk_form() -> None
```

### `scripts/`
```python
profile_raw.py    main()   # -> docs/data_quality_report.md
run_etl.py        main()   # argparse --rebuild --verify
train_all.py      main()   # argparse --model {all,outcome,fare,customer,driver} --tune
make_insights.py  main()   # -> docs/INSIGHTS.md from queries + metrics
```

### `tests/`
```python
test_cleaning.py   # datetime parsing, outcome-null preservation, FK integrity
test_features.py   # rush-hour flag, city_pair, leave-one-out history correctness
test_dataset.py    # assert_no_leakage raises on every blocked column
test_queries.py    # each query returns non-empty with the expected columns
test_models.py     # trained model beats baseline; artefact round-trips
```

---

## 6. Day-by-Day Plan

| Day | Deliverable | Definition of done |
|---|---|---|
| 1 | `config.py`, `io.py`, `profile_raw.py` | **DONE** - report written, 13 tests green |
| 2 | `cleaning.py` + tests | **DONE** - 0 rows lost, structural nulls preserved |
| 3 | `schema.py`, `db.py`, `etl.py` | **DONE** - 141,959 rows in MySQL, 7 indexes, all PASS |
| 4 | `features.py` + tests | **DONE** - 76-column table cached; 38 tests green |
| 5 | `01_eda.ipynb`, `stats.py`, `charts.py` | all 7 spec EDA outputs + 3 significance tests |
| 6 | `dataset.py`, `pipeline.py`, models 1 and 2 | leaderboards vs baseline; leakage ablation recorded |
| 7 | models 3 and 4, `explain.py`, `registry.py` | 4 artefacts + `metrics.json` |
| 8 | `app.py` + 6 analytics pages | filters work end-to-end, cached, paginated |
| 9 | `model_lab.py`, `predict.py` | live prediction on all 4 models |
| 10 | `README.md`, `INSIGHTS.md`, slides, final test run | demo walkthrough rehearsed |

**Optional (spec Step 6):** FastAPI `/predict/*` endpoints and a drift-monitoring page — worth doing on day 10 only if days 1–9 landed clean.

---

## 7. Marks Mapping

| Metric | Where it's earned |
|---|---|
| Code Quality / Transformations (10) | `cleaning.py` + `features.py`, PEP 8, docstrings, try/except on DB and model I/O |
| Documentation (10) | `README.md`, `INSIGHTS.md`, `data_quality_report.md`, this plan, slides |
| Code Reusability (10) | `rapido/` package; zero logic in `app_pages/`; `tests/` |
| Presentation (10) | Capstone guideline order: domain → problem → cleaning → EDA → FE → stats → imbalance → base model → models → metric → final model → importance → business action |
| Task Accomplishment (10) | 4 models + 20 queries + 8 pages + SQL layer |
| Mock Q&A (10) | Leakage story (§1.2), metric choice, index justification, SMOTE vs class-weight comparison |

---

## 8. Business Outputs to State Explicitly

Peak cancellation windows (city × hour) · high-risk ride flagging at booking time · driver allocation rule derived from the reliability score · pre-quote fare accuracy against the surge formula · the ops intervention that delivers the spec's 20% cancellation reduction.
