-- Rapido Intelligent Mobility Insights - MySQL schema
--
-- Normalised star schema: three generated dimensions (cities, vehicle_types,
-- locations), three source dimensions (customers, drivers, time_features) and
-- two fact tables (location_demand, bookings).
--
-- This file is the single source of truth for the schema. src/data_preprocessing.py
-- reads and executes it directly, so the DDL here and the DDL that runs are
-- always the same statements.
--
-- Apply with:  python src/data_preprocessing.py etl --rebuild

CREATE DATABASE IF NOT EXISTS rapido_mobility
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE rapido_mobility;

-- ---------------------------------------------------------------------------
-- Tables, in foreign-key-safe creation order
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cities (
    city_id   SMALLINT     NOT NULL AUTO_INCREMENT,
    city_name VARCHAR(50)  NOT NULL,
    PRIMARY KEY (city_id),
    UNIQUE KEY uq_cities_name (city_name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS vehicle_types (
    vehicle_type_id SMALLINT    NOT NULL AUTO_INCREMENT,
    vehicle_name    VARCHAR(30) NOT NULL,
    PRIMARY KEY (vehicle_type_id),
    UNIQUE KEY uq_vehicle_name (vehicle_name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS locations (
    location_id   INT         NOT NULL AUTO_INCREMENT,
    city_id       SMALLINT    NOT NULL,
    location_code VARCHAR(20) NOT NULL,
    PRIMARY KEY (location_id),
    UNIQUE KEY uq_location_city_code (city_id, location_code),
    CONSTRAINT fk_locations_city FOREIGN KEY (city_id)
        REFERENCES cities (city_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS customers (
    customer_id               VARCHAR(12)  NOT NULL,
    customer_gender           VARCHAR(20),
    customer_age              TINYINT UNSIGNED,
    city_id                   SMALLINT,
    customer_signup_days_ago  SMALLINT UNSIGNED,
    preferred_vehicle_type_id SMALLINT,
    total_bookings            SMALLINT UNSIGNED,
    completed_rides           SMALLINT UNSIGNED,
    cancelled_rides           SMALLINT UNSIGNED,
    incomplete_rides          SMALLINT UNSIGNED,
    cancellation_rate         DECIMAL(6,4),
    avg_customer_rating       DECIMAL(3,2),
    customer_cancel_flag      TINYINT(1),
    PRIMARY KEY (customer_id),
    CONSTRAINT fk_customers_city FOREIGN KEY (city_id)
        REFERENCES cities (city_id),
    CONSTRAINT fk_customers_vehicle FOREIGN KEY (preferred_vehicle_type_id)
        REFERENCES vehicle_types (vehicle_type_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS drivers (
    driver_id               VARCHAR(12) NOT NULL,
    driver_age              TINYINT UNSIGNED,
    city_id                 SMALLINT,
    vehicle_type_id         SMALLINT,
    driver_experience_years TINYINT UNSIGNED,
    total_assigned_rides    SMALLINT UNSIGNED,
    accepted_rides          SMALLINT UNSIGNED,
    incomplete_rides        SMALLINT UNSIGNED,
    delay_count             SMALLINT UNSIGNED,
    acceptance_rate         DECIMAL(6,4),
    delay_rate              DECIMAL(6,4),
    avg_driver_rating       DECIMAL(3,2),
    avg_pickup_delay_min    DECIMAL(6,2),
    driver_delay_flag       TINYINT(1),
    PRIMARY KEY (driver_id),
    CONSTRAINT fk_drivers_city FOREIGN KEY (city_id)
        REFERENCES cities (city_id),
    CONSTRAINT fk_drivers_vehicle FOREIGN KEY (vehicle_type_id)
        REFERENCES vehicle_types (vehicle_type_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS time_features (
    slot_datetime  DATETIME    NOT NULL,
    hour_of_day    TINYINT UNSIGNED NOT NULL,
    day_of_week    VARCHAR(10) NOT NULL,
    is_weekend     TINYINT(1)  NOT NULL,
    is_holiday     TINYINT(1)  NOT NULL,
    peak_time_flag TINYINT(1)  NOT NULL,
    season         VARCHAR(20) NOT NULL,
    PRIMARY KEY (slot_datetime)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS location_demand (
    demand_id            INT      NOT NULL AUTO_INCREMENT,
    city_id              SMALLINT NOT NULL,
    location_id          INT      NOT NULL,
    hour_of_day          TINYINT UNSIGNED NOT NULL,
    vehicle_type_id      SMALLINT NOT NULL,
    total_requests       INT UNSIGNED,
    completed_rides      INT UNSIGNED,
    cancelled_rides      INT UNSIGNED,
    avg_wait_time_min    DECIMAL(8,3),
    avg_surge_multiplier DECIMAL(4,2),
    demand_level         ENUM('Low','Medium','High'),
    PRIMARY KEY (demand_id),
    UNIQUE KEY uq_demand_slot (location_id, hour_of_day, vehicle_type_id),
    CONSTRAINT fk_demand_city FOREIGN KEY (city_id)
        REFERENCES cities (city_id),
    CONSTRAINT fk_demand_location FOREIGN KEY (location_id)
        REFERENCES locations (location_id),
    CONSTRAINT fk_demand_vehicle FOREIGN KEY (vehicle_type_id)
        REFERENCES vehicle_types (vehicle_type_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS bookings (
    booking_id              VARCHAR(12) NOT NULL,
    booking_ts              DATETIME    NOT NULL,
    city_id                 SMALLINT    NOT NULL,
    pickup_location_id      INT         NOT NULL,
    drop_location_id        INT         NOT NULL,
    vehicle_type_id         SMALLINT    NOT NULL,
    customer_id             VARCHAR(12) NOT NULL,
    driver_id               VARCHAR(12) NOT NULL,
    ride_distance_km        DECIMAL(6,2),
    estimated_ride_time_min DECIMAL(6,2),
    actual_ride_time_min    DECIMAL(6,2) NULL,
    traffic_level           ENUM('Low','Medium','High'),
    weather_condition       ENUM('Clear','Rain','Heavy Rain'),
    base_fare               DECIMAL(8,2),
    surge_multiplier        DECIMAL(4,2),
    booking_value           DECIMAL(10,2),
    booking_status          ENUM('Completed','Cancelled','Incomplete') NOT NULL,
    incomplete_ride_reason  VARCHAR(50) NULL,
    PRIMARY KEY (booking_id),
    CONSTRAINT fk_bookings_city FOREIGN KEY (city_id)
        REFERENCES cities (city_id),
    CONSTRAINT fk_bookings_pickup FOREIGN KEY (pickup_location_id)
        REFERENCES locations (location_id),
    CONSTRAINT fk_bookings_drop FOREIGN KEY (drop_location_id)
        REFERENCES locations (location_id),
    CONSTRAINT fk_bookings_vehicle FOREIGN KEY (vehicle_type_id)
        REFERENCES vehicle_types (vehicle_type_id),
    CONSTRAINT fk_bookings_customer FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id),
    CONSTRAINT fk_bookings_driver FOREIGN KEY (driver_id)
        REFERENCES drivers (driver_id)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- Indexes
--
-- Each index is paired with the dashboard query that justifies it; none are
-- speculative.
-- ---------------------------------------------------------------------------

-- Date-range filter applied on every dashboard page.
CREATE INDEX idx_bookings_ts ON bookings (booking_ts);

-- Cancellation rate by city; the composite avoids a full scan per city.
CREATE INDEX idx_bookings_city_status ON bookings (city_id, booking_status);

-- High-risk customer drill-down and leave-one-out history aggregates.
CREATE INDEX idx_bookings_customer ON bookings (customer_id);

-- Driver reliability leaderboard.
CREATE INDEX idx_bookings_driver ON bookings (driver_id);

-- Cancellation split by vehicle type.
CREATE INDEX idx_bookings_vehicle_status ON bookings (vehicle_type_id, booking_status);

-- Top pickup locations and route-pair aggregation.
CREATE INDEX idx_bookings_pickup ON bookings (pickup_location_id);

-- Hourly demand heatmap.
CREATE INDEX idx_demand_city_hour ON location_demand (city_id, hour_of_day);
