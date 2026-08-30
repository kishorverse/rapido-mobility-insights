# Data Quality Report

Generated from `Rapido_dataset/` (5 source files).

## 1. Structure

| dataset         |   rows |   columns |
|:----------------|-------:|----------:|
| bookings        | 100000 |        22 |
| customers       |  10000 |        13 |
| drivers         |   5000 |        14 |
| location_demand |  17941 |        10 |
| time_features   |   8760 |         7 |

Booking date range: **2025-01-01** to **2025-12-31**.

## 2. Target Distribution

| booking_status   |   rows |   share_pct |
|:-----------------|-------:|------------:|
| Completed        |  68346 |       68.35 |
| Cancelled        |  23284 |       23.28 |
| Incomplete       |   8370 |        8.37 |

## 3. Missing Values

| dataset   | column                 |   nulls |   null_pct |
|:----------|:-----------------------|--------:|-----------:|
| bookings  | actual_ride_time_min   |   31654 |      31.65 |
| bookings  | incomplete_ride_reason |   91630 |      91.63 |

Missingness here is **structural, not random** - see section 6.

## 4. Duplicate Keys

| dataset       | key         |   duplicates | status   |
|:--------------|:------------|-------------:|:---------|
| bookings      | booking_id  |            0 | PASS     |
| customers     | customer_id |            0 | PASS     |
| drivers       | driver_id   |            0 | PASS     |
| time_features | datetime    |            0 | PASS     |

## 5. Referential Integrity

| check                             |   orphan_rows | status   |
|:----------------------------------|--------------:|:---------|
| bookings.customer_id -> customers |             0 | PASS     |
| bookings.driver_id -> drivers     |             0 | PASS     |

## 6. Leakage Check 1 - Post-Outcome Columns

| booking_status   |   actual_ride_time_null_rate |   incomplete_reason_present_rate |
|:-----------------|-----------------------------:|---------------------------------:|
| Cancelled        |                            1 |                                0 |
| Completed        |                            0 |                                0 |
| Incomplete       |                            1 |                                1 |

`actual_ride_time_min` is null for **every** non-Completed ride, so its null indicator alone reproduces the target. Both columns are excluded from all model feature matrices.

## 7. Leakage Check 2 - Fare Is a Formula

`booking_value / (base_fare * surge_multiplier)`:

|   mean_ratio |   std_ratio |   min_ratio |   max_ratio |
|-------------:|------------:|------------:|------------:|
|      0.99993 |      0.0289 |     0.94989 |     1.05003 |

The ratio sits in [0.94989, 1.05003] with standard deviation 0.0289. Fare is `base_fare * surge` plus roughly 5% noise, so `base_fare` is excluded from the pre-quote fare model.

## 8. Leakage Check 3 - Dimension Flags Are Thresholds

| dataset   | flag                 |   flag_value | rate_column       |   rate_min |   rate_max |   rows |
|:----------|:---------------------|-------------:|:------------------|-----------:|-----------:|-------:|
| customers | customer_cancel_flag |            0 | cancellation_rate |     0      |       0.2  |   4657 |
| customers | customer_cancel_flag |            1 | cancellation_rate |     0.2105 |       1    |   5343 |
| drivers   | driver_delay_flag    |            0 | delay_rate        |     0      |       0.1  |   4346 |
| drivers   | driver_delay_flag    |            1 | delay_rate        |     0.11   |       0.42 |    654 |

The flags are exact cut-offs on their own rate column, so they cannot serve as model targets. Booking-level outcomes are used instead.

## 9. Zero-Variance Columns

| dataset       | column     |   distinct_values | value   |
|:--------------|:-----------|------------------:|:--------|
| time_features | is_holiday |                 1 | [0]     |

## 10. Full Column Profile

| dataset         | column                   | dtype    |   non_null |   nulls |   null_pct |   unique |    min |     max |    mean | sample_values                                                                                           |
|:----------------|:-------------------------|:---------|-----------:|--------:|-----------:|---------:|-------:|--------:|--------:|:--------------------------------------------------------------------------------------------------------|
| bookings        | booking_id               | string   |     100000 |       0 |       0    |   100000 | nan    |  nan    | nan     | B_000001, B_000002, B_000003, B_000004, B_000005                                                        |
| bookings        | booking_date             | str      |     100000 |       0 |       0    |      365 | nan    |  nan    | nan     | 2025-12-11, 2025-07-07, 2025-08-23, 2025-04-12, 2025-11-22                                              |
| bookings        | booking_time             | str      |     100000 |       0 |       0    |     1440 | nan    |  nan    | nan     | 00:07:00, 06:13:00, 08:53:00, 10:25:00, 00:08:00                                                        |
| bookings        | day_of_week              | category |     100000 |       0 |       0    |        7 | nan    |  nan    | nan     | Thursday, Monday, Saturday, Tuesday, Sunday                                                             |
| bookings        | is_weekend               | int64    |     100000 |       0 |       0    |        2 |   0    |    1    |   0.284 | nan                                                                                                     |
| bookings        | hour_of_day              | int64    |     100000 |       0 |       0    |       24 |   0    |   23    |  11.506 | nan                                                                                                     |
| bookings        | city                     | category |     100000 |       0 |       0    |        5 | nan    |  nan    | nan     | Mumbai, Chennai, Delhi, Hyderabad, Bangalore                                                            |
| bookings        | pickup_location          | str      |     100000 |       0 |       0    |       50 | nan    |  nan    | nan     | Loc_19, Loc_32, Loc_28, Loc_16, Loc_22                                                                  |
| bookings        | drop_location            | str      |     100000 |       0 |       0    |       50 | nan    |  nan    | nan     | Loc_16, Loc_38, Loc_1, Loc_30, Loc_31                                                                   |
| bookings        | vehicle_type             | category |     100000 |       0 |       0    |        3 | nan    |  nan    | nan     | Bike, Cab, Auto                                                                                         |
| bookings        | ride_distance_km         | float64  |     100000 |       0 |       0    |     2401 |   1    |   25    |  13.028 | nan                                                                                                     |
| bookings        | estimated_ride_time_min  | float64  |     100000 |       0 |       0    |    15591 |   3    |  164.98 |  61.384 | nan                                                                                                     |
| bookings        | actual_ride_time_min     | float64  |      68346 |   31654 |      31.65 |    15370 |   2.75 |  197.34 |  61.447 | nan                                                                                                     |
| bookings        | traffic_level            | category |     100000 |       0 |       0    |        3 | nan    |  nan    | nan     | High, Medium, Low                                                                                       |
| bookings        | weather_condition        | category |     100000 |       0 |       0    |        3 | nan    |  nan    | nan     | Heavy Rain, Rain, Clear                                                                                 |
| bookings        | base_fare                | float64  |     100000 |       0 |       0    |    38720 |  28.02 |  529.96 | 211.678 | nan                                                                                                     |
| bookings        | surge_multiplier         | float64  |     100000 |       0 |       0    |       12 |   1    |    2.3  |   1.589 | nan                                                                                                     |
| bookings        | booking_value            | float64  |     100000 |       0 |       0    |    52651 |  27.28 | 1265.59 | 336.345 | nan                                                                                                     |
| bookings        | booking_status           | category |     100000 |       0 |       0    |        3 | nan    |  nan    | nan     | Cancelled, Completed, Incomplete                                                                        |
| bookings        | incomplete_ride_reason   | str      |       8370 |   91630 |      91.63 |        4 | nan    |  nan    | nan     | Driver Delay, App Issue, Customer No-show, Vehicle Issue                                                |
| bookings        | customer_id              | string   |     100000 |       0 |       0    |    10000 | nan    |  nan    | nan     | C_005097, C_008459, C_003471, C_002161, C_005617                                                        |
| bookings        | driver_id                | string   |     100000 |       0 |       0    |     5000 | nan    |  nan    | nan     | D_004592, D_000148, D_004976, D_001173, D_001175                                                        |
| customers       | customer_id              | string   |      10000 |       0 |       0    |    10000 | nan    |  nan    | nan     | C_000001, C_000002, C_000003, C_000004, C_000005                                                        |
| customers       | customer_gender          | category |      10000 |       0 |       0    |        3 | nan    |  nan    | nan     | Non-Binary, Male, Female                                                                                |
| customers       | customer_age             | int64    |      10000 |       0 |       0    |       47 |  18    |   64    |  41.054 | nan                                                                                                     |
| customers       | customer_city            | category |      10000 |       0 |       0    |        5 | nan    |  nan    | nan     | Bangalore, Delhi, Hyderabad, Chennai, Mumbai                                                            |
| customers       | customer_signup_days_ago | int64    |      10000 |       0 |       0    |      970 |  30    |  999    | 517.501 | nan                                                                                                     |
| customers       | preferred_vehicle_type   | category |      10000 |       0 |       0    |        3 | nan    |  nan    | nan     | Cab, Bike, Auto                                                                                         |
| customers       | total_bookings           | int64    |      10000 |       0 |       0    |       24 |   1    |   26    |  10     | nan                                                                                                     |
| customers       | completed_rides          | int64    |      10000 |       0 |       0    |       21 |   0    |   20    |   6.835 | nan                                                                                                     |
| customers       | cancelled_rides          | int64    |      10000 |       0 |       0    |       11 |   0    |   10    |   2.328 | nan                                                                                                     |
| customers       | incomplete_rides         | int64    |      10000 |       0 |       0    |        6 |   0    |    5    |   0.837 | nan                                                                                                     |
| customers       | cancellation_rate        | float64  |      10000 |       0 |       0    |       92 |   0    |    1    |   0.233 | nan                                                                                                     |
| customers       | avg_customer_rating      | float64  |      10000 |       0 |       0    |       16 |   3.5  |    5    |   4.252 | nan                                                                                                     |
| customers       | customer_cancel_flag     | int64    |      10000 |       0 |       0    |        2 |   0    |    1    |   0.534 | nan                                                                                                     |
| drivers         | driver_id                | string   |       5000 |       0 |       0    |     5000 | nan    |  nan    | nan     | D_000001, D_000002, D_000003, D_000004, D_000005                                                        |
| drivers         | driver_age               | int64    |       5000 |       0 |       0    |       33 |  22    |   54    |  37.776 | nan                                                                                                     |
| drivers         | driver_city              | category |       5000 |       0 |       0    |        5 | nan    |  nan    | nan     | Bangalore, Chennai, Mumbai, Delhi, Hyderabad                                                            |
| drivers         | vehicle_type             | category |       5000 |       0 |       0    |        3 | nan    |  nan    | nan     | Auto, Cab, Bike                                                                                         |
| drivers         | driver_experience_years  | int64    |       5000 |       0 |       0    |       14 |   1    |   14    |   7.485 | nan                                                                                                     |
| drivers         | total_assigned_rides     | int64    |       5000 |       0 |       0    |       32 |   6    |   38    |  20     | nan                                                                                                     |
| drivers         | accepted_rides           | int64    |       5000 |       0 |       0    |       28 |   3    |   30    |  15.343 | nan                                                                                                     |
| drivers         | incomplete_rides         | int64    |       5000 |       0 |       0    |        8 |   0    |    7    |   1.674 | nan                                                                                                     |
| drivers         | delay_count              | int64    |       5000 |       0 |       0    |        7 |   0    |    6    |   0.946 | nan                                                                                                     |
| drivers         | acceptance_rate          | float64  |       5000 |       0 |       0    |       58 |   0.31 |    1    |   0.767 | nan                                                                                                     |
| drivers         | delay_rate               | float64  |       5000 |       0 |       0    |       28 |   0    |    0.42 |   0.048 | nan                                                                                                     |
| drivers         | avg_driver_rating        | float64  |       5000 |       0 |       0    |       11 |   4    |    5    |   4.493 | nan                                                                                                     |
| drivers         | avg_pickup_delay_min     | float64  |       5000 |       0 |       0    |       55 |   1    |   10.3  |   3.231 | nan                                                                                                     |
| drivers         | driver_delay_flag        | int64    |       5000 |       0 |       0    |        2 |   0    |    1    |   0.131 | nan                                                                                                     |
| location_demand | city                     | category |      17941 |       0 |       0    |        5 | nan    |  nan    | nan     | Bangalore, Chennai, Delhi, Hyderabad, Mumbai                                                            |
| location_demand | pickup_location          | str      |      17941 |       0 |       0    |       50 | nan    |  nan    | nan     | Loc_1, Loc_10, Loc_11, Loc_12, Loc_13                                                                   |
| location_demand | hour_of_day              | int64    |      17941 |       0 |       0    |       24 |   0    |   23    |  11.499 | nan                                                                                                     |
| location_demand | vehicle_type             | category |      17941 |       0 |       0    |        3 | nan    |  nan    | nan     | Auto, Bike, Cab                                                                                         |
| location_demand | total_requests           | int64    |      17941 |       0 |       0    |       15 |   1    |   15    |   5.574 | nan                                                                                                     |
| location_demand | completed_rides          | int64    |      17941 |       0 |       0    |       14 |   0    |   13    |   3.809 | nan                                                                                                     |
| location_demand | cancelled_rides          | int64    |      17941 |       0 |       0    |        9 |   0    |    8    |   1.298 | nan                                                                                                     |
| location_demand | avg_wait_time_min        | float64  |      17941 |       0 |       0    |    16497 |   3.74 |  164.19 |  61.399 | nan                                                                                                     |
| location_demand | avg_surge_multiplier     | float64  |      17941 |       0 |       0    |      527 |   1    |    2.3  |   1.589 | nan                                                                                                     |
| location_demand | demand_level             | category |      17941 |       0 |       0    |        2 | nan    |  nan    | nan     | Low, Medium                                                                                             |
| time_features   | datetime                 | str      |       8760 |       0 |       0    |     8760 | nan    |  nan    | nan     | 2025-01-01 00:00:00, 2025-01-01 01:00:00, 2025-01-01 02:00:00, 2025-01-01 03:00:00, 2025-01-01 04:00:00 |
| time_features   | hour_of_day              | int64    |       8760 |       0 |       0    |       24 |   0    |   23    |  11.5   | nan                                                                                                     |
| time_features   | day_of_week              | category |       8760 |       0 |       0    |        7 | nan    |  nan    | nan     | Wednesday, Thursday, Friday, Saturday, Sunday                                                           |
| time_features   | is_weekend               | int64    |       8760 |       0 |       0    |        2 |   0    |    1    |   0.285 | nan                                                                                                     |
| time_features   | is_holiday               | int64    |       8760 |       0 |       0    |        1 |   0    |    0    |   0     | nan                                                                                                     |
| time_features   | peak_time_flag           | int64    |       8760 |       0 |       0    |        2 |   0    |    1    |   0.292 | nan                                                                                                     |
| time_features   | season                   | category |       8760 |       0 |       0    |        3 | nan    |  nan    | nan     | Winter, Summer, Monsoon                                                                                 |
