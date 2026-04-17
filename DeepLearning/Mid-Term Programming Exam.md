# National Chung Cheng University

## Introduction to Deep Learning  
**Mid-Term Programming Exam (Take-Home)**  
Period: 2026-04-23 ~ 2026-05-07 (deadline: 23:59:59) — 50% of total

---

## Overview
You are provided a time-series forecasting problem centered on restaurant visitors. The data comes from two separate systems:

- Hot Pepper Gourmet (hpg): similar to Yelp — users can search restaurants and make online reservations.  
- AirREGI / Restaurant Board (air): reservation control and cash register system.

Use reservations, visits, and other information to forecast future restaurant visitor totals on given dates. The dataset covers 2016 until April 2017. Split data into:
- Training: full year of 2016
- Testing: the data provided for 2017

Notes:
- Some days in the test set have restaurants closed (no visitors); these days are ignored in scoring.
- The training set omits days when restaurants were closed.

---

## File descriptions
Files are prefixed with `air_` or `hpg_` to indicate source. Each restaurant has a unique `air_store_id` and `hpg_store_id`. Not all restaurants appear in both systems. Latitude/longitude are approximate.

### Reservation files
- `air_reserve.csv`  
    Contains reservations made in the air system.  
    Fields:
    - `air_store_id` — restaurant id in air system
    - `visit_datetime` — datetime of the reservation (visit date/time)
    - `reserve_datetime` — datetime when reservation was created
    - `reserve_visitors` — number of visitors for that reservation

- `hpg_reserve.csv`  
    Contains reservations made in the hpg system.  
    Fields:
    - `hpg_store_id` — restaurant id in hpg system
    - `visit_datetime`
    - `reserve_datetime`
    - `reserve_visitors`

### Store info files
- `air_store_info.csv`  
    Information about select air restaurants. Columns:
    - `air_store_id`
    - `air_genre_name`
    - `air_area_name`
    - `latitude`
    - `longitude`  
    Note: latitude/longitude correspond to the area the store belongs to (approximate).

- `hpg_store_info.csv`  
    Information about select hpg restaurants. Columns:
    - `hpg_store_id`
    - `hpg_genre_name`
    - `hpg_area_name`
    - `latitude`
    - `longitude`  
    Note: latitude/longitude correspond to the area the store belongs to (approximate).

### Mapping and visit data
- `store_id_relation.csv`  
    Mapping between systems for stores that appear in both:
    - `hpg_store_id`
    - `air_store_id`

- `air_visit_data.csv`  
    Historical visit data for air restaurants. Fields:
    - `air_store_id`
    - `visit_date` — the date
    - `visitors` — number of visitors on that date

---

## Submission
Submit your model code and fill in the following information in your report:

Model Information (Final Selected Model: ResNet1D):
- Model name: ResNet1D
- Number of layers: 8
- Number of units in each layer: Stem(Conv64) + ResidualBlocks(64,128,128) + GAP + FC(1)
- Activation functions used: ReLU + residual skip connections + Softplus output
- Loss function (for ResNet1D training):
    - Mean Squared Error (MSE)
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
- Cost function (for ResNet1D optimization target):
    - Same as loss in this project (epoch-level average MSE)
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Evaluation metric required by exam:
    - RMSLE = sqrt((1/N) * sum_{i=1..N} (log(1 + y_i) - log(1 + y_hat_i))^2)
- Training epochs: 100
- Training accuracy: N/A (project metric uses RMSLE; Train R2 = 63.36%)
- Testing accuracy: N/A (project metric uses RMSLE; Test R2 = -97.40%)
- Optimization techniques employed:
        1. Train-only feature scaling + full timeline reindex preprocessing
        2. Feature engineering (visitors lag/rolling features)
        3. Optimization and model architecture upgrades (AdamW + scheduler + Softplus + ResNet1D)

Model Information (Baseline Comparison Model: MLP):
- Model name: MLP
- Number of layers: 2
- Number of units in each layer: input_dim -> 64 -> 1
- Activation functions used: ReLU (hidden) + Softplus output
- Loss function (for MLP training):
    - Mean Squared Error (MSE)
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
- Cost function (for MLP optimization target):
    - Same as loss in this project (epoch-level average MSE)
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Evaluation metric required by exam:
    - RMSLE = sqrt((1/N) * sum_{i=1..N} (log(1 + y_i) - log(1 + y_hat_i))^2)

Difference in accuracies after each optimization technique applied:
1) Optimization technique name: Feature engineering (lag/rolling)  
    - Before optimization: Training/Testing Accuracies = N/A / N/A (RMSLE workflow)  
    - After optimization: Training/Testing Accuracies = N/A / N/A (RMSLE workflow)  
    - Any other changes: Added `visitors_lag_1/7/14`, `visitors_roll_mean_7/std_7`; feature dimension from 9 to 14.

2) Optimization technique name: Training stability upgrades  
    - Before optimization: Training/Testing Accuracies = N/A / N/A (RMSLE workflow)  
    - After optimization: Training/Testing Accuracies = N/A / N/A (RMSLE workflow)  
    - Any other changes: Switched to AdamW + ReduceLROnPlateau, output activation to Softplus, target transform to log1p.

3) Optimization technique name: Final model selection (MLP vs ResNet1D)  
    - Before optimization: Training/Testing Accuracies = Train/Test RMSLE = 0.578145 / 0.578952 (MLP baseline)  
    - After optimization: Training/Testing Accuracies = Train/Test RMSLE = 0.558909 / 0.560029 (ResNet1D final)  
    - Any other changes: ResNet1D selected as final model; overfit gap remains small (0.001120, Overfitting Risk = No).

Anything special about your model:  
Final pipeline focuses on reproducibility and exam alignment: test scoring excludes closed days (`visitors=0`), preprocessing avoids leakage, and the training flow is locked to a stable two-model comparison (`mlp`, `resnet1d`).

Comments on the course:  
The take-home format was practical and helped connect preprocessing quality, model architecture, and metric alignment in a real forecasting workflow.

