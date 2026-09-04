# 🛒 Rossmann Store Sales Analysis & Forecasting

## Project Overview

This project explores historical sales data from **Rossmann stores** to identify patterns affecting store sales and customer traffic and to build a time-series forecasting model for future sales.

The analysis covers **data cleaning, exploratory data analysis (EDA), data visualization, feature engineering, correlation analysis, and sales forecasting using Prophet**.

The original datasets contain more than **1 million daily observations across 1,115 stores**, along with store-level information such as promotions, competition, store type, and assortment.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Explore historical store sales and customer behavior
* Clean and preprocess sales and store information
* Investigate missing values and store characteristics
* Analyze relationships between sales, customers, promotions, and other variables
* Examine sales patterns across months, weekdays, and store types
* Evaluate the impact of promotions on sales and customer traffic
* Engineer date-based features for analysis
* Build a time-series model to forecast future sales for individual stores

---

## 📊 Dataset

The project uses the **Rossmann Store Sales** dataset from Kaggle.

Two main datasets are used:

### `train.csv`

Contains daily historical information for each store, including:

* `Store` - unique store Id
* `Sales` - sales/day, this is the target variable
* `Customers` - the number of customers on a given day
* `Open` - an indicator for whether the store was open: 0 = closed, 1 = open
* `Promo` - indicates whether a store is running a promo on that day
* `StateHoliday` - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* `SchoolHoliday` - indicates if the (Store, Date) was affected by the closure of public schools


### `store.csv`

Contains store-level characteristics, including:

* `Store` - a unique Id for each store
* `StoreType` — Store model/type
* `Assortment` — describes an assortment level: a = basic, b = extra, c = extended
* `CompetitionDistance` — Distance to the nearest competitor
* `CompetitionOpenSinceMonth`- gives the approximate month of the time the nearest competitor was opened
* `CompetitionOpenSinceYear` - gives the approximate year of the time the nearest competitor was opened
* `Promo` - indicates whether a store is running a promo on that day
* `Promo2` — Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
* `Promo2SinceWeek` - describes the calendar week when the store started participating in Promo2
* `Promo2SinceYear` - describes the year week when the store started participating in Promo2
* `PromoInterval` - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand the structure and behavior of the data.

### Missing Values

Missing values in the store dataset are investigated and handled.

For example:

* Missing promotion-related values are replaced where appropriate
* Missing competition opening information is handled
* Missing `CompetitionDistance` values are filled using the mean competition distance

### Store Status

Closed stores are removed from the sales analysis so that the analysis focuses on days when stores were actually operating.

### Distribution Analysis

Histograms are used to explore the distributions of variables such as:

* Sales
* Customers
* Store activity
* Promotions
* School holidays
* Competition information

---

## 🔗 Combining Sales and Store Data

The sales and store datasets are merged using the unique `Store` identifier.

```python
sales_store_df = pd.merge(
    sales_train_df,
    store_df,
    on='Store',
    how='inner'
)
```

This creates a combined dataset containing both daily sales information and store characteristics.

---

## 📈 Correlation Analysis

A correlation matrix is generated to investigate relationships between numerical variables.

Some of the strongest relationships observed in the analysis include:

* **Sales and Customers** — strong positive relationship
* **Sales and Promo** — moderate positive relationship
* **Sales and Promo2** — relatively weak relationship

A heatmap is used to visualize these correlations.

---

## 📅 Feature Engineering

The `Date` variable is transformed into additional time-based features:

```python
sales_store_df['Year'] = pd.DatetimeIndex(sales_store_df['Date']).year
sales_store_df['Month'] = pd.DatetimeIndex(sales_store_df['Date']).month
sales_store_df['Day'] = pd.DatetimeIndex(sales_store_df['Date']).day
```

These features make it possible to analyze sales patterns over different time periods.

---

## 📊 Sales & Customer Analysis

The project examines average sales and customer traffic across:

* Months
* Days of the month
* Days of the week
* Years
* Store types

The analysis also compares sales and customer behavior when promotions are active versus inactive.

Both **bar plots** and **violin plots** are used to explore the effect of promotions.

---

## 🏪 Store Type Analysis

Sales trends are compared across different Rossmann store types.

Daily average sales are grouped by `Date` and `StoreType` to visualize how the different store categories perform over time.

---

## 🔮 Sales Forecasting with Prophet

The project uses **Prophet** for time-series forecasting.

A reusable forecasting function is created that allows sales predictions to be generated for a selected store.

```python
def sales_forcast(Store_ID, sales_df, periods):

    sales_df = sales_df[sales_df['Store'] == Store_ID]

    sales_df = sales_df.rename(
        columns={'Date': 'ds', 'Sales': 'y'}
    )

    sales_df = sales_df.sort_values(by='ds')

    model = Prophet()
    model.fit(sales_df)

    future = model.make_future_dataframe(periods=periods)

    forecast = model.predict(future)

    model.plot(forecast)
    model.plot_components(forecast)
```

For example:

```python
sales_forcast(10, sales_train_df, 60)
```

forecasts sales for **Store 10 for the next 60 days**.

Prophet also allows the forecast to be decomposed into components to better understand underlying time-series patterns.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Prophet**
* **Jupyter Notebook**

---

## 📂 Project Structure

```text
Rossmann-Sales-Forecasting/
│
├── Sales_Department.ipynb
├── train.csv
├── store.csv
├── sales_store_df.csv
└── README.md
```

## 💡 Key Takeaways

The analysis demonstrates how historical retail data can be transformed into useful business insights and forecasting models.

Key observations from the project include:

* Customer traffic is strongly associated with sales.
* Promotions are associated with higher sales and customer activity.
* Sales behavior varies across time periods and store types.
* Time-series forecasting can be used to estimate future sales at the individual-store level.


---

## 📌 Data Source

**Rossmann Store Sales — Kaggle Competition**

The dataset contains historical sales data for Rossmann drug stores and is intended for forecasting daily sales.
* `Data Source`: https://www.kaggle.com/c/rossmann-store-sales/data
---


