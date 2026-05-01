# Retail store demand forecaster

Predicting daily sales for 50 items across 10 stores using the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) dataset on Kaggle.

---

## Overview

The goal is simple: given 5 years of past sales, predict what will sell each day for the next 3 months.

Two models were tested. Linear Regression is the baseline, it is the simplest model possible and shows what you get without any complexity. LightGBM is the main model, it is faster, more accurate, and better at finding patterns in real-world data.

LightGBM scored 12.78% SMAPE on 2017 data it had never seen during training. Lower is better, anything under 15% is considered good for this dataset.

---

## Data

Two files are needed. Both come from Kaggle.

| File | Rows | Columns | Notes |
|---|---|---|---|
| train.csv | 913,000 | 4 | Has sales values. Used for training. |
| test.csv | 45,000 | 4 | No sales values. Predict these. |

The four columns in train.csv are: `date`, `store`, `item`, `sales`.
test.csv replaces `sales` with `id`.

The data was split by year:
- Train: 2013–2016 (730,500 rows)
- Validation: 2017 (182,500 rows)
- Test: January–March 2018 (45,000 rows)

---

## Cleaning

The data is clean overall. One problem was found and fixed.

**The zero sales row**:
During the cleaning process there was one store that had zero sales. Every other store had normal sales that day, so the zero is a data entry error. It was replaced with the average sales of the day before and the day after.

**Outliers**:
11,967 rows were flagged by the IQR method as unusually high sales. These were kept. In retail, high sales days are common; factors such as promotions, holidays, or bulk orders are usually the cause. Removing them would make the model worse at predicting those days.

**Rescaling**:
Rescaling was not applied. LightGBM splits data by asking if a value is above or below a certain threshold, so it doesn't care if one feature ranges from 1–10 and another from 1–365. I also skipped rescaling for Linear Regression since the feature ranges were similar enough to not cause any issues.

**One-hot encoding**:
Store and item are category labels, not real numbers. Store 10 is not "10 times more" than store 1, they are just different locations. Linear Regression treats everything as a number, so store and item were one-hot encoded for that model using `pd.get_dummies()`. This adds 60 new columns,one per store and one per item each containing a 0 or 1. LightGBM handles category labels on its own, so no encoding was needed there.

---

## What the data looks like

Sales were split into four groups based on daily volume:

| Class | Sales range | Row count |
|---|---|---|
| Low | 0–30 | 240,211 |
| Medium-Low | 31–47 | 222,985 |
| Medium-High | 48–70 | 228,871 |
| High | 71+ | 220,933 |

Each group has roughly the same number of rows, which makes comparisons fair.

Five charts compare how each feature distributes across the four groups:
<img width="1520" height="570" alt="Image" src="https://github.com/user-attachments/assets/9ef9d5f2-d43e-49f4-bf56-46eb9379db89" />

<img width="1516" height="570" alt="Image" src="https://github.com/user-attachments/assets/bdbc39b3-ac24-4b51-aa9d-2008c39e62f6" />

<img width="1490" height="576" alt="Image" src="https://github.com/user-attachments/assets/14199b09-85ad-4ce9-94f7-56b59e5b57e7" />

<img width="1494" height="572" alt="Image" src="https://github.com/user-attachments/assets/3746eacd-8daa-4c81-9fcd-0093dabaa6a5" />

<img width="1512" height="574" alt="Image" src="https://github.com/user-attachments/assets/4ed0159a-057f-4fb1-bd12-a5cd51255dd5" />

**Month**: The clearest pattern. High sales bunch up in June, July, and August. Low sales dominate in January and February. Month is the strongest single predictor.

**Day of week**: Small but consistent differences across groups. Certain days lean toward higher sales than others.

**Year**: The High group grows bigger each year from 2013 to 2017. Overall demand went up year over year.

**Item**: Some items land in the High group more often. The pattern exists but is weaker than the seasonal one.

**Store**: Sales spread fairly evenly across all groups for every store. Store ID alone does not tell you much.

---

## Models

**Input features:** store, item, year, month, day_of_week

**Target:** daily sales count

**Metric:** SMAPE (Symmetric Mean Absolute Percentage Error). It measures the average percentage difference between what the model predicted and what actually happened. Lower is better.

---

### Linear Regression

The simplest model. It draws a straight line through the data and uses that to predict. It cannot learn that sales are high on summer weekends at specific stores, it can only fit one flat relationship per feature.

Store and item were one-hot encoded before training so the model treats them as categories rather than numbers.

---

### LightGBM

Builds 200 small decision trees one at a time. Each tree fixes the mistakes of the one before it. After 200 rounds the trees vote together to make a prediction.

Settings used:
- `n_estimators = 200`
- `learning_rate = 0.1` 
- `random_state = 42`

---

## Results

| Model | SMAPE | Notes |
|---|---|---|
| Linear Regression | 22.53% | Baseline. One-hot encoded. |
| LightGBM | 12.78% | Main model. No encoding needed. |

<img width="1222" height="880" alt="Image" src="https://github.com/user-attachments/assets/592b8c0c-f935-4a34-8ae9-faf646f8353a" />

<img width="1224" height="874" alt="Image" src="https://github.com/user-attachments/assets/36cf9949-49cd-4f41-b435-9daa9ff02c32" />

LightGBM is 9.75 percentage points better. The gap comes from LightGBM's ability to learn combinations of features, something Linear Regression cannot do. LightGBM was used to generate the final Kaggle submission.

---

## Conclusions

A five-feature model (store, item, year, month, day of week) is enough to score 12.78% SMAPE. Month is the most useful feature, summer sales are consistently higher than winter sales. LightGBM is significantly better than Linear Regression for this type of problem because retail demand does not follow a straight line.

---

## How to run this

### Step 1 — Install packages

```bash
pip install pandas numpy matplotlib lightgbm scikit-learn
```

Install this first or lightgbm will not load.

```bash
brew install libomp
pip install lightgbm
```

### Step 2 — Get the data

1. Go to [https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
2. Sign in to Kaggle
3. Click **Download All**
4. Put `train.csv` and `test.csv` in the same folder as the notebook

---

## Files in this repository

```
├── README.md                        — Overview
├── retail_demand_forecasting.ipynb  — Main notebook. All steps in one place:
│                                      cleaning, visualization,
│                                      model training, evaluation.

```

---

## Packages used

| Package | What it does |
|---|---|
| pandas | Loads and manipulates the data |
| numpy | Math operations used in the SMAPE function |
| matplotlib | Draws all the charts |
| lightgbm | Trains the LightGBM model |
| scikit-learn | Trains the Linear Regression model |
