**![][image1]**

| Project Title | Rapido: Intelligent Mobility Insights: Ride Patterns, Cancellations & Fare Forecasting |
| :---- | :---- |
| **Skills take away From This Project** | **Python scripting, Data Cleaning, Machine Learning, Data Management using SQL, Streamlit** |
| **Domain** | **Mobility & Transportation Analytics** |

**Problem Statement:**

Rapido operates a large-scale ride-hailing platform where millions of bookings are created daily across multiple cities, vehicle types, and demand conditions. A significant challenge for the business is **ride cancellations, inaccurate fare estimation, inefficient driver allocation, and poor customer experience during peak demand**.

Despite collecting rich trip-level data (bookings, customers, drivers, locations, and time signals), these insights are not fully leveraged to **predict booking outcomes, optimize pricing, and proactively manage operational risks**.

The objective of this project is to **build a unified Machine Learning–driven decision system** using realistic, large-scale booking data to:

* Predict ride outcomes before trip start

* Estimate accurate fares dynamically

* Identify high-risk customers and drivers

* Enable data-driven operational interventions

**Business Use Cases:**

### **Use Case 1 — Reduce Cancellations by 20%**

### **Use Case 2 — Improve ETA Accuracy**

### **Use Case 3 — Dynamic Pricing (Demand Prediction)**

### **Use Case 4 — Driver Reliability Scoring**

**Approach:**

1) **Ride Outcome Prediction (Multi-Class Classification)**  
    Predict whether a booking will be:

* Completed

* Cancelled

* Incomplete

2) **Fare Prediction Model (Regression)**  
    Predict the expected booking value prior to trip confirmation based on:

* Distance

* Traffic & weather

* Time of day

* Vehicle type

* Surge dynamics

3) **Customer Cancellation Risk Model (Binary Classification)**  
    Predict the probability that a customer will cancel a booking using:

* Historical cancellation rate

* Past ratings

* Peak-time behavior

* Pricing sensitivity

4) **Driver Delay Prediction Model (Binary Classification)**  
    Predict whether a driver is likely to cause delays or incomplete rides based on:

* Past delay history

* Traffic exposure

* Acceptance behavior

  #  **End-to-End Approach :**

  # **Step 1 — Data Cleaning**

* Fix missing values

* Convert time, date to datetime

* Create new features

* Encode categorical columns

  ---

  ### **Step 2 — Exploratory Data Analysis**

You should include:

* Ride volume by hour, weekday, city

* Cancellation heatmap across cities

* Distance vs Fare correlation

* Rating distribution

* Customer vs Driver behaviour comparison

* Payment method usage patterns

* Traffic/Weather vs Cancellation  
  ---

  ### **Step 3 — Feature Engineering**

Create new features, such as:

* `Fare_per_KM`

* `Fare_per_Min`

* `Rush_Hour_Flag`

* `Long_Distance_Flag`

* `City_Pair` \= Pickup \+ Drop

* `Driver_Reliability_Score`

* `Customer_Loyalty_Score`

  ### **Step 4 — Model Training**

  Split into train/test (80/20).  
   Tune using GridSearch/Optuna.  
  ---

  ### **Step 5 — Model Evaluation**

  For classification:  
* Accuracy

* F1-score

* AUC

* Confusion matrix

  For regression:  
* RMSE

* MAE

* R²

  Target Industry benchmarks:  
* Classification Accuracy → **85–90%**

* Regression RMSE → **within ±10% of actual fare**

  ---

  ### **Step 6 — Deployment** 

* Streamlit dashboard

* Prediction API (FastAPI/Flask) **(Optional)**

* Model monitoring dashboard **(Optional)**  
  ---

  #  

  # **6\. Expected Output**

  ### **Business-Level Outputs**

* Identify peak cancellation windows

* Predict high-risk rides

* Suggest driver allocation strategy

* Estimate fare more accurately

* Improve ops decision-making

  ### **Model Outputs**

* A trained Cancellation Prediction Model

* Fare Prediction Model

* Feature importance ranking

* Interactive dashboards (Streamlit)

  ### **Visualization Outputs**

* Pickup/Drop city heatmaps

* Cancellations by hour

* Surge behavior patterns

* Customer vs Driver cancellation reasons

**DataSet:**  
	  
**[Rapido\_dataset](https://drive.google.com/drive/folders/1ZmESmEXCoVYzep1hXNagAE7wD3g-7uI1?usp=sharing)**

| File | Description |
| ----- | ----- |
| `bookings.csv` | Core transactional data with booking outcomes and fare targets |
| `customers.csv` | Customer behavior and historical cancellation signals |
| `drivers.csv` | Driver performance, delay, and reliability metrics |
| `location_demand.csv` | Aggregated demand patterns by location & time |
| `time_features.csv` | Enriched temporal signals (hour, weekday, seasonality, peaks) |

**Project Guidelines:**

1. **Coding Standards**  
   * Use meaningful names: Variables, functions, and database tables should have descriptive names.  
   * Follow PEP 8 (for Python): Maintain consistent formatting with proper indentation and spacing.  
   * Modularize your code: Break your code into functions or classes to enhance readability and reusability.  
   * Error handling: Implement try-except blocks for handling API errors and SQL exceptions.  
   * Document your code: Include docstrings and comments to explain logic and functions.  
2. **SQL Database Practices**  
   * Normalize tables: Avoid redundancy and ensure efficient data storage.  
   * Use indexes: Optimize query performance with appropriate indexing.  
   * Follow naming conventions: Use consistent and descriptive names for tables and fields.  
3.  **Streamlit Application Development**  
   * Interactive features: Ensure the UI is responsive, with interactive widgets for filters.  
   * Minimalist design: Keep the layout simple for a smooth user experience.  
   * Performance optimization: Avoid loading all data at once—use pagination or batch processing where possible.  
4. **General Best Practices**  
   * Test frequently: Regularly test each component (e.g., API requests, SQL queries, Streamlit app) during development.  
   * Backup your data: Maintain backups of your SQL database and code.  
   * Documentation: Provide a README file with setup instructions, project objectives, and a demo walkthrough.

**Reference:**

***\*\*If you don’t know how to approach the project, kindly refer to the project orientation recording provided in this table. (Available in English & Tamil)***

| Streamlit Doc | [https://docs.streamlit.io/library/api-reference](https://docs.streamlit.io/library/api-reference) |
| :---- | :---- |
| **Project Live Evaluation Metrics** | [Project Live Evaluation](https://docs.google.com/document/u/0/d/1QisLD2kqDWFZJG2oDknKn2eMGi-Xq8oFPgA7UWSbcIQ/edit) |
| **Capstone Explanation Guideline** | [Capstone Explanation Guideline](https://docs.google.com/document/d/1gbhLvJYY7J73lu1g9c6C9LRJvYemiDOdRDAEMe632w8/edit) |
| **GitHub Reference** | [How to Use GitHub.pptx](https://docs.google.com/presentation/d/1XHCbgUOqbcXNUyQ87vTlKdKRgAbBxtkA/edit?usp=sharing&ouid=109735616107417446342&rtpof=true&sd=true) |
| **Project Orientation (English)** |  **[Project Orientation Session : ML Project (English)](https://docs.google.com/document/d/1IyC1iCy62V9E7zdbpOBbw0wqMG2dU-Q_r2GZTg25RkY/edit?usp=sharing)** |

**Timeline:**

The project must be completed and submitted within **10 days** from the assigned date.

**PROJECT DOUBT CLARIFICATION SESSION ( PROJECT AND CLASS DOUBTS)**

**About Session:** The Project Doubt Clarification Session is a helpful resource for resolving questions and concerns about projects and class topics. It provides support in understanding project requirements, addressing code issues, and clarifying class concepts. The session aims to enhance comprehension and provide guidance to overcome challenges effectively.  
**Note: Book the slot at least before 12:00 Pm on the same day**

**Timing: Monday to Saturday (4:00PM to 5:00PM)**

**Booking link :[https://forms.gle/XC553oSbMJ2Gcfug9](https://forms.gle/XC553oSbMJ2Gcfug9)**

**LIVE EVALUATION SESSION (CAPSTONE AND FINAL PROJECT)**

**About Session:** The Live Evaluation Session for Capstone and Final Projects allows participants to showcase their projects and receive real-time feedback for improvement. It assesses project quality and provides an opportunity for discussion and evaluation.  
**Note: This form will Open on Saturday and Sunday Only on Every Week**

**Timing: Monday-Saturday (5:30 PM to 6:30 PM)**

**Booking link : [https://forms.gle/1m2Gsro41fLtZurRA](https://forms.gle/1m2Gsro41fLtZurRA)**  
      


[image1]: <data:image/png;base64,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>