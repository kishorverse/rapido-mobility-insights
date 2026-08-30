# Rapido: Intelligent Mobility Insights
### Ride Patterns, Cancellations & Fare Forecasting

An end-to-end analytics and machine-learning system over 100,000 Rapido bookings:
Python data pipeline → normalised MySQL warehouse → four ML models → interactive
Streamlit dashboard.

**Domain:** Mobility & Transportation Analytics
**Stack:** Python · pandas · scikit-learn · MySQL · Streamlit · Plotly

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Dataset](#4-dataset)
5. [Data Cleaning](#5-data-cleaning)
6. [Database Design](#6-database-design)
7. [Exploratory Findings](#7-exploratory-findings)
8. [Feature Engineering](#8-feature-engineering)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Leakage Controls](#10-leakage-controls)
11. [Dashboard Walkthrough](#11-dashboard-walkthrough)
12. [Testing](#12-testing)
13. [Business Recommendations](#13-business-recommendations)
14. [Documented Deviations](#14-documented-deviations)
15. [Project Structure](#15-project-structure)

---

## 1. Objectives

- Predict a ride's outcome **before the trip starts** (Completed / Cancelled / Incomplete)
- Estimate the fare **before booking confirmation**
- Score a customer's probability of cancelling
- Score a driver's probability of causing a delay or incomplete ride
- Surface demand and cancellation patterns through an interactive dashboard

---

## 2. Quick Start

### Prerequisites
- Python 3.11+
- MySQL 8.0+ running locally

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure database credentials
cp .env.example .env        # then edit .env with your MySQL password

# 3. Profile the raw data  ->  docs/data_quality_report.md
python scripts/profile_raw.py

# 4. Build the warehouse (creates schema, loads 141,959 rows, builds indexes)
python scripts/run_etl.py --rebuild

# 5. Train the four models  ->  models/*.joblib
python scripts/train_all.py

# 6. Generate the findings report  ->  docs/INSIGHTS.md
python scripts/make_insights.py

# 7. Launch the dashboard
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

### Configuration

All settings live in `config.py` and are overridable through environment variables
(loaded from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `RAPIDO_DB_HOST` | `localhost` | MySQL host |
| `RAPIDO_DB_PORT` | `3306` | MySQL port |
| `RAPIDO_DB_USER` | `root` | MySQL user |
| `RAPIDO_DB_PASSWORD` | *(empty)* | MySQL password |
| `RAPIDO_DB_NAME` | `rapido_mobility` | Database name |

`.env` is gitignored — no credential is ever committed.

### Other commands

```bash
python scripts/run_etl.py --verify-only        # connectivity + row counts
python scripts/train_all.py --model fare --tune # retrain one model with search
python scripts/train_all.py --tune --search grid  # exhaustive GridSearchCV sweep
python scripts/build_notebook.py --execute      # regenerate the EDA notebook
pytest tests -q                                 # run the test suite
```

---

## 3. Architecture

```
Rapido_dataset/*.csv
        │
        ▼
   rapido/io.py ──────────► scripts/profile_raw.py ──► docs/data_quality_report.md
        │
        ▼
   rapido/cleaning.py          (validate, parse, preserve structural nulls)
        │
        ├──────────────► rapido/features.py ──► data/processed/features.parquet
        │                                              │
        ▼                                              ▼
   rapido/etl.py                              rapido/models/
        │                                     ├── dataset.py   (leakage guard)
        ▼                                     ├── pipeline.py  (preprocess + estimator)
   MySQL: rapido_mobility                     ├── train.py     (4 entry points)
   (8 tables, 3NF, 7 indexes)                 ├── evaluate.py  (metrics)
        │                                     ├── explain.py   (importance)
        ▼                                     ├── registry.py  (persistence)
   rapido/queries.py  (30 named queries)      └── serve.py     (prediction)
        │                                              │
        └──────────────┬───────────────────────────────┘
                       ▼
              app.py + app_pages/   (9-page Streamlit dashboard)
```

**Design rule:** all business logic lives in the `rapido/` package. The Streamlit pages
under `app_pages/` only orchestrate and render — they contain no queries, no feature
maths, and no model code. This keeps every capability testable and reusable outside the
dashboard.

---

## 4. Dataset

| File | Rows | Grain | Role |
|---|---|---|---|
| `bookings.csv` | 100,000 | one booking | fact table, all four targets |
| `customers.csv` | 10,000 | one customer | customer dimension |
| `drivers.csv` | 5,000 | one driver | driver dimension |
| `location_demand.csv` | 17,941 | city × zone × hour × vehicle | demand aggregate |
| `time_features.csv` | 8,760 | one hour of 2025 | calendar dimension |

Coverage: 2025-01-01 to 2025-12-31 · 5 cities · 3 vehicle types · 50 zones per city.

**Target distribution:** Completed 68,346 (68.3%) · Cancelled 23,284 (23.3%) ·
Incomplete 8,370 (8.4%)

---

## 5. Data Cleaning

Handled in `rapido/cleaning.py`:

- **Column standardisation** to snake_case, whitespace stripped
- **Datetime parsing** — `booking_date` + `booking_time` collapse into one `booking_ts`
- **Type coercion** — explicit numeric and categorical dtypes
- **Duplicate removal** by primary key (none found — all keys unique)
- **Referential integrity validation** (no orphan foreign keys found)
- **Range validation** against documented plausible bounds — all columns PASS
- **Outlier detection** by IQR, reported rather than capped
- **Location namespacing** — `Loc_1..Loc_50` repeat in every city, so codes are
  city-qualified into 250 distinct zones

### Missing values: a deliberate decision

The only nulls in the dataset are **structural**:

| Column | Nulls | Cause |
|---|---|---|
| `actual_ride_time_min` | 31,654 | null for exactly every non-Completed ride |
| `incomplete_ride_reason` | 91,630 | present only for the 8,370 Incomplete rides |

These are **not imputed**. Filling them would fabricate the very signal the models are
meant to predict. `handle_missing_bookings()` explicitly skips
`config.POST_OUTCOME_COLUMNS`, and both columns are blocked from every feature matrix.

Result: 0 rows lost, 0 non-structural nulls, all range checks PASS.

---

## 6. Database Design

Eight tables in third normal form:

```
cities ─┬─ locations ─┬─ location_demand
        ├─ customers  │
        ├─ drivers    │
        └─────────────┴─ bookings ─── vehicle_types
                         time_features
```

**Normalisation decisions:**

- `city`, `location` and `vehicle_type` are repeated text in the CSVs → surrogate-keyed
  dimensions.
- `locations` is keyed on **(city_id, location_code)**, because `Loc_1..Loc_50` repeat
  across all five cities. Without this the dimension would collapse five cities' zones
  into one.
- `day_of_week`, `is_weekend` and `hour_of_day` are **dropped from `bookings`** — they
  are transitively dependent on `booking_ts` and already exist in `time_features`.
  Keeping them would violate 3NF.
- `traffic_level`, `weather_condition`, `booking_status` and `demand_level` stay as
  `ENUM` columns: closed low-cardinality domains carrying no attributes of their own, so
  a lookup table would add a join without removing redundancy.

**Indexes** — each exists to serve a specific dashboard query:

| Index | Table | Serves |
|---|---|---|
| `idx_bookings_ts` | bookings | date-range filter on every page |
| `idx_bookings_city_status` | bookings | cancellation rate by city |
| `idx_bookings_customer` | bookings | customer drill-down |
| `idx_bookings_driver` | bookings | driver leaderboard |
| `idx_bookings_vehicle_status` | bookings | cancellation by vehicle type |
| `idx_bookings_pickup` | bookings | top pickup zones, route aggregation |
| `idx_demand_city_hour` | location_demand | hourly demand heatmap |

Load: **141,959 rows across 8 tables in ~22 seconds**, every table verified PASS.

---

## 7. Exploratory Findings

Full report: [`docs/INSIGHTS.md`](docs/INSIGHTS.md) · Notebook:
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

### Traffic and weather drive cancellations — geography does not

| Traffic | Cancel % | Incomplete % |
|---|---|---|
| High | **33.50** | **14.82** |
| Low | 18.26 | 5.14 |
| Medium | 17.93 | 5.05 |

| Weather | Cancel % | Incomplete % |
|---|---|---|
| Heavy Rain | **33.67** | 8.40 |
| Rain | 26.04 | 8.43 |
| Clear | 10.01 | 8.28 |

Meanwhile cancellation rate by city spans only **22.95% – 23.78%**, and by vehicle type
only **23.17% – 23.41%**.

### Surge is the strongest single lever

| Surge band | Cancel % |
|---|---|
| None (1.0x) | 5.25 |
| Low (1.0–1.5x) | 11.27 |
| Medium (1.5–2.0x) | 34.15 |
| High (>2.0x) | **35.25** |

A near sevenfold increase — and surge is platform-controlled.

### Two distinct failure modes

Weather changes the cancellation rate by 24 points but leaves the **incompletion rate
flat at 8.3–8.4%**. Traffic changes both. Riders abandon bookings in bad weather;
drivers fail to complete them in bad traffic. These need different interventions.

### Accountability is overwhelmingly driver-side

Incomplete rides carry a stated reason. Attributing each reason to the party that can
act on it (`queries.REASON_PARTY`) splits the 8,370 incomplete rides as:

| Party | Reasons | Rides | Share |
|---|---|---|---|
| **Driver** | Driver Delay (4,728), Vehicle Issue (1,265) | **5,993** | **71.6%** |
| Platform | App Issue | 1,221 | 14.6% |
| Customer | Customer No-show | 1,156 | 13.8% |

Driver Delay alone is 56.5% of all incompletions. This is why the driver-risk model
targets incompletion rather than cancellation, and why the recommended interventions
in section 13 are weighted toward supply-side coaching rather than rider penalties.

### Statistical significance

| Test | Variables | Effect size | Conclusion |
|---|---|---|---|
| Chi-square | traffic × outcome | V = **0.185** | significant |
| Chi-square | weather × outcome | V = **0.168** | significant |
| Chi-square | vehicle × outcome | V = 0.003 | **not significant** (p = 0.70) |
| Chi-square | city × outcome | V = 0.007 | **not significant** (p = 0.40) |
| One-way ANOVA | fare across vehicle types | F = 27,763 | significant |
| Welch's t-test | fare weekend vs weekday | d = -0.001 | **not significant** (p = 0.92) |

**Why these tests:** chi-square for categorical-vs-categorical association; ANOVA to
compare three vehicle-type means in one test without inflating family-wise error;
Welch's t-test for two groups without assuming equal variances. Cramér's V accompanies
every chi-square because at n = 100,000 p-values collapse toward zero — effect size is
what separates a real driver from a trivial one.

---

## 8. Feature Engineering

The master table carries **76 columns**. Engineered features split into two groups.

**Context features** (from the booking row):
`rush_hour_flag` · `is_night_ride` · `long_distance_flag` · `distance_band` ·
`city_route_pair` · `is_same_zone` · `expected_speed_kmph` · `surge_bucket` ·
`adverse_conditions_flag` · `bad_weather_flag` · `high_traffic_flag` ·
`fare_per_km` · `fare_per_min` · `time_of_day_band`

**Scores:**
- `driver_reliability_score` (0–100) — acceptance 35%, punctuality 35%, rating 30%
- `customer_loyalty_score` (0–100) — volume 40% (log-scaled), completion 40%, rating 20%

**Prior-history features** — the important ones:

`cust_prior_rides` · `cust_prior_cancel_rate` · `cust_prior_completion_rate` ·
`drv_prior_rides` · `drv_prior_incomplete_rate` · `*_is_first_ride` (and more)

These are built as an **expanding window over strictly earlier bookings**, so a booking
never contributes to its own predictors. The static `cancellation_rate` and `delay_rate`
columns shipped in the dimension files are whole-period aggregates that already include
the row being predicted — they are replaced, not used.

**The proof:** `cust_prior_cancel_rate` has exactly **10,000 nulls** — one per
customer's first-ever ride — and `drv_prior_cancel_rate` exactly **5,000**. If the
current row were leaking in, those counts would be zero. Locked in by
`tests/test_features.py`.

---

## 9. Machine Learning Models

Protocol for all four: leakage-checked dataset → stratified 80/20 split → baseline
(`DummyClassifier` / mean) → four candidate estimators → 5-fold CV → held-out evaluation
→ permutation importance → persist.

### Results (held-out test set, 20,000 rows)

| Model | Type | Algorithm | Headline | Secondary |
|---|---|---|---|---|
| Ride Outcome | 3-class | HistGradientBoosting | F1-macro **0.5652** | ROC-AUC (OvR) 0.8193 |
| Fare Prediction | regression | HistGradientBoosting | R² **0.9966** | MAPE 2.76%, within ±10%: **99.82%** |
| Cancellation Risk | binary | HistGradientBoosting | ROC-AUC **0.8512** | PR-AUC 0.6422 |
| Driver Delay Risk | binary | RandomForest | ROC-AUC **0.7235** | PR-AUC 0.2360 |

### On the project's accuracy benchmark

The brief targets 85–90% classification accuracy. **That is not reachable from pre-trip
signal on this data, and any submission reporting it is reading a post-outcome column.**

- 68.3% of bookings complete, so guessing the majority class already scores 68.3%.
- Balanced class weights were used deliberately, trading headline accuracy for recall on
  the 8.4% Incomplete class — the class operations actually cares about. Macro-F1 rose
  from the 0.478 baseline to 0.5652, and balanced accuracy sits at 0.6185 against 0.333
  for random.
- ROC-AUC (OvR) of 0.8193 shows genuine three-class discrimination.

The fare benchmark **is** met, and then some: 99.82% of predictions fall within ±10%.

### Why these metrics

- **Macro-F1** leads the outcome model — plain accuracy is inflated by the majority class.
- **PR-AUC** accompanies ROC-AUC on the risk models — ROC-AUC flatters imbalanced positives.
- **±10% hit rate** leads the fare model, because the brief states its target as a
  tolerance band.

---

## 10. Leakage Controls

Three traps in this dataset produce spectacular but meaningless scores. Each is
identified, controlled, and covered by a test.

| # | Trap | Evidence | Control |
|---|---|---|---|
| 1 | `actual_ride_time_min`, `incomplete_ride_reason` | null for **100%** of Cancelled/Incomplete and **0%** of Completed — the null indicator alone reproduces the target | blocked from all four models |
| 2 | `base_fare` | `booking_value / (base_fare × surge)` ∈ [0.950, 1.050], sd 0.029 | blocked from the fare model |
| 3 | `cancellation_rate`, `delay_rate`, and the `*_flag` columns | whole-period aggregates including the predicted row; the flags are those rates thresholded at exactly 0.20 and 0.10 | replaced with prior-history features |

`rapido/models/dataset.py::assert_no_leakage()` **raises** rather than warns, and
`tests/test_dataset.py` fails if any blocked column reappears.

### The fare model is at its noise floor, not overfitting

The tariff was recovered exactly by linear fit (**R² = 1.000000** per vehicle type):

| Vehicle | Flagfall | Per km |
|---|---|---|
| Bike | ₹20 | ₹8 |
| Auto | ₹40 | ₹12 |
| Cab | ₹80 | ₹18 |

`booking_value = base_fare × surge × (1 ± 5% uniform noise)`.

A uniform variable on [0.95, 1.05] has mean absolute deviation of **2.50%**, so **no
model can beat 2.50% MAPE on this data**. The trained model reaches **2.76%** — the
noise floor, not leakage. A regression test asserts MAPE stays ≥ 2.4%, because dropping
below the floor would itself prove leakage.

**Ablation:** refitting *with* `base_fare` moves R² only from 0.9966 to 0.9968,
confirming the tariff is already recoverable from distance and vehicle type.

---

## 11. Dashboard Walkthrough

`streamlit run app.py` — nine pages under two sections.

**Analytics**
| Page | Contents |
|---|---|
| **Overview** | KPI cards, hourly/city volume, monthly trend, key findings, recommended actions |
| **Demand & Volume** | Hour/weekday/month patterns, day×hour heatmap, top zones, busiest routes, zone wait times |
| **Cancellations** | Condition drivers, outcome-mix stacked bars, surge bands, city×hour heatmap, peak windows, reason breakdown |
| **Fares & Revenue** | Interactive tariff calculator, distance-fare scatter, fare distributions, surge curve, revenue treemap |
| **Customers** | Demographic segments, high-risk table with progress bars, rating distributions |
| **Drivers** | Reliability scatter, delay-risk table, top-performer leaderboard, allocation rule |
| **Data Explorer** | Server-side paginated booking records (50/page across 2,000 pages) |

**Machine Learning**
| Page | Contents |
|---|---|
| **Model Lab** | Portfolio table, per-model leaderboards vs baseline, CV results, confusion matrices, ROC/PR curves, threshold trade-off table, permutation importance, leakage explainer + ablation |
| **Live Prediction** | One form scoring all four models — outcome with class probabilities, fare with confidence band vs tariff, two risk gauges, and a recommended operational action |

Global sidebar filters (city, vehicle, traffic, weather, date range, hour range) apply
across every analytics page. All queries are cached with a TTL and bounded entry count;
the explorer pages in SQL rather than loading the full table.

---

## 12. Testing

```bash
pytest tests -q
```

**145 tests, all passing.**

| File | Covers |
|---|---|
| `test_io.py` | loading, dtypes, uniqueness, referential integrity, the three leakage assumptions |
| `test_cleaning.py` | datetime parsing, structural-null preservation, outliers, ranges, namespacing |
| `test_features.py` | engineered flags, scores, and prior-history correctness on a hand-built fixture |
| `test_dataset.py` | the leakage guard, parameterised across every blocked column × every target |
| `test_queries.py` | all 30 queries against live MySQL, parameterisation, SQL-injection safety, pagination |
| `test_models.py` | metric maths, artefact round-trips, baseline comparison, benchmark and noise-floor assertions, serving behaviour |

Database-dependent tests skip automatically when MySQL is unreachable.

---

## 13. Business Recommendations

1. **Cap surge during adverse conditions.** Cancellations run 5.3% at no surge and 35.3%
   above 2.0x, and heavy rain plus high traffic is where risk compounds. Surge is
   platform-controlled — the most directly actionable lever in the data.
2. **Allocate on traffic, not on city.** Cancellation rates differ by under one point
   across the five cities; traffic nearly doubles them. Allocation should key on live
   traffic and zone demand pressure.
3. **Score bookings before dispatch.** ROC-AUC 0.851 on pre-trip information only. The
   Model Lab threshold table lets operations choose a cut-off matching their intervention
   capacity.
4. **Treat the two failure modes separately.** Weather-driven cancellations need
   rider-side intervention (fare guarantees, ETA transparency). Traffic-driven
   incompletions need driver-side routing support.
5. **Prioritise reliable drivers in high-traffic windows.** Driver Delay is the largest
   single incomplete-ride reason (~4,700 of 8,370), and traffic raises the incompletion
   rate from 5.1% to 14.8%.

---

## 14. Documented Deviations

Four requirements in the brief cannot be met as literally stated. Each is substituted
with the nearest defensible equivalent.

| Requested | Reality | Substitution |
|---|---|---|
| Payment-method usage analysis | **No payment column exists** in any of the five source files | Vehicle-type × surge usage patterns |
| `City_Pair = Pickup City + Drop City` | Every booking is **intra-city**, and `Loc_1..Loc_50` repeat identically across all five cities | `city_route_pair` = city + pickup + drop; locations keyed on (city, code) |
| `is_holiday` feature | **0 for all 8,760 hours** of 2025 — zero variance | dropped via `config.ZERO_VARIANCE_COLUMNS` |
| 85–90% classification accuracy | Majority class alone scores 68.3%; reaching 90% requires a post-outcome column | honest metrics reported with the leakage evidence and ablation |

---

## 15. Project Structure

```
Project_3/
├── app.py                      # Streamlit entry point and navigation
├── config.py                   # paths, DB config, constants, leakage blocklists
├── requirements.txt
├── .env.example                # credential template (.env is gitignored)
│
├── rapido/                     # all business logic
│   ├── io.py                   # loading and Parquet caching
│   ├── cleaning.py             # validation and cleaning
│   ├── features.py             # feature engineering
│   ├── schema.py               # DDL and index definitions
│   ├── db.py                   # MySQL access layer
│   ├── etl.py                  # extract-transform-load pipeline
│   ├── queries.py              # 30 named parameterised queries
│   ├── charts.py               # Plotly figure builders
│   ├── stats.py                # significance testing
│   └── models/
│       ├── dataset.py          # feature matrices + leakage guard
│       ├── pipeline.py         # preprocessing + estimators
│       ├── train.py            # training entry points
│       ├── evaluate.py         # metrics and diagnostics
│       ├── explain.py          # feature importance
│       ├── registry.py         # artefact persistence
│       └── serve.py            # prediction serving
│
├── app_pages/                  # 9 Streamlit pages (presentation only)
├── scripts/                    # profile_raw · run_etl · train_all · make_insights · build_notebook
├── tests/                      # 145 tests
├── notebooks/01_eda.ipynb      # executed EDA notebook
├── docs/
│   ├── PROJECT_BRIEF.md        # the original project specification
│   ├── PROJECT_PLAN.md         # build plan and function inventory
│   ├── data_quality_report.md  # generated profiling report
│   └── INSIGHTS.md             # generated findings report
├── data/processed/             # Parquet caches (gitignored)
├── models/                     # trained .joblib artefacts (gitignored)
└── Rapido_dataset/             # source CSVs
```

---

## Author

Built as a capstone project for the GUVI Data Science programme.
