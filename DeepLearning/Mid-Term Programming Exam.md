# National Chung Cheng University

## Introduction to Deep Learning  
**Mid-Term Programming Exam (Take-Home)**  
Period: 2026-04-23 ~ 2026-05-07 (deadline: 23:59:59) - 50% of total

---

## Overview
You are provided a time-series forecasting problem centered on restaurant visitors. The data comes from two separate systems:

- Hot Pepper Gourmet (hpg): similar to Yelp - users can search restaurants and make online reservations.  
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
    - `air_store_id` - restaurant id in air system
    - `visit_datetime` - datetime of the reservation (visit date/time)
    - `reserve_datetime` - datetime when reservation was created
    - `reserve_visitors` - number of visitors for that reservation

- `hpg_reserve.csv`  
    Contains reservations made in the hpg system.  
    Fields:
    - `hpg_store_id` - restaurant id in hpg system
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
    - `visit_date` - the date
    - `visitors` - number of visitors on that date

---

## Submission
Submit your model code and fill in the following information in your report:

Evaluation metrics used in this project (three separate metrics):
- Accuracy (%): exact-match rate after rounding to integer visitors
    - Accuracy = (1/N) * sum_{i=1..N} I(round(max(y_hat_i,0)) = round(max(y_i,0))) * 100
- R2 (coefficient of determination):
    - R2 = 1 - [sum_{i=1..N}(y_i - y_hat_i)^2] / [sum_{i=1..N}(y_i - y_bar)^2]
- RMSLE (official exam metric):
    - RMSLE = sqrt((1/N) * sum_{i=1..N} (log(1 + y_i) - log(1 + y_hat_i))^2)

Model Information (基準比較模型: MLP)
- Model name: MLP
- Number of layers: 2
- Number of units in each layer: input_dim -> 64 -> 1
- Activation functions used: ReLU (hidden) + Softplus output
- Loss / Cost function (training objective):
    - Mean Squared Error (MSE)
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Training epochs: 100

Model Information (最終選擇模型: ResNet1D)
- Model name: ResNet1D
- Number of layers: 8
- Number of units in each layer: Stem(Conv64) + ResidualBlocks(64,128,128) + GAP + FC(1)
- Activation functions used: ReLU + residual skip connections + Softplus output
- Loss / Cost function (training objective):
    - Mean Squared Error (MSE)
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Training epochs: 100

Training / Testing summary (latest regenerated run from results.csv and best_model_summary.md):
- MLP (baseline)
    - Train Accuracy: 37.85%
    - Test Accuracy: 5.49%
    - Train R2: 0.587871
    - Test R2: 0.364594
    - Train RMSLE: 0.578145
    - Test RMSLE: 0.578952
    - Train RMSE: 10.484785
    - Test RMSE: 13.593495
    - Train MAE: 5.363681
    - Test MAE: 8.150771

- ResNet1D (final selected model)
    - Train Accuracy: 38.08%
    - Test Accuracy: 5.87%
    - Train R2: 0.633555
    - Test R2: -0.973978
    - Train RMSLE: 0.558909
    - Test RMSLE: 0.560029
    - Train RMSE: 9.927783
    - Test RMSE: 23.959431
    - Train MAE: 5.131245
    - Test MAE: 7.896169
    - Test Peak Recall: 0.446825
    - Overfit Gap (Test-Train RMSLE): 0.001120 (Risk: No)

Difference in accuracies / metrics after each optimization technique applied:
1) Optimization technique name: 前處理管線重構（reindex + train-only scaling）
    - Before optimization (historical pipeline): Accuracy and generalization were unstable due to discontinuous per-store timelines and potential preprocessing leakage.
    - After optimization: model behavior became stable and reproducible under fixed data semantics.
    - Any other changes:
        - per-store reindex + 店休日 visitors 補 0
        - 類別 mapping 與標準化只用訓練資料建立（train-only）
        - 測試評估排除 visitors=0（符合題目 scoring 規則）

2) Optimization technique name: 特徵工程升級（lag/rolling）
    - Before optimization: features were mainly static/calendar features.
    - After optimization: temporal dependency modeling improved with lag/rolling features.
    - Any other changes:
        - 新增 visitors_lag_1 / visitors_lag_7 / visitors_lag_14
        - 新增 visitors_roll_mean_7 / visitors_roll_std_7
        - 全部以 shift(1) 實作，避免使用當日資訊造成 leakage
        - 特徵維度由 9 增加到 14

3) Optimization technique name: 訓練穩定化與最終模型篩選（MLP vs ResNet1D）
    - Before optimization (MLP baseline):
        - Train Accuracy / Test Accuracy = 37.85% / 5.49%
        - Train R2 / Test R2 = 0.587871 / 0.364594
        - Train RMSLE / Test RMSLE = 0.578145 / 0.578952
    - After optimization (ResNet1D final):
        - Train Accuracy / Test Accuracy = 38.08% / 5.87%
        - Train R2 / Test R2 = 0.633555 / -0.973978
        - Train RMSLE / Test RMSLE = 0.558909 / 0.560029
    - Any other changes:
        - 優化器改 AdamW，加入 ReduceLROnPlateau
        - target transform 採 log1p，推論端以 expm1 還原
        - 輸出層使用 Softplus，確保輸出非負
        - 固定 100 epochs，關閉 early stopping

Anything special about your model:
本專案的重點不是盲目加深模型，而是讓資料流程、評估規則與模型學習空間完全對齊：
1. 資料前處理管線（對齊 preprocessing.py）
    - per-store reindex + 店休日補 0
    - leakage-safe lag_1/7/14 + roll_mean_7/std_7（皆採 shift(1)）
    - train-only StandardScaler + train-only 類別 mapping
2. 模型架構細節（對齊 models_training.py）
    - ResNet1D: Stem(Conv1d 64) + 3 個 ResidualBlock1D（含 skip connection）+ AdaptiveAvgPool1d + Softplus
    - MLP: input_dim -> 64 -> 1（baseline）
    - peak-weighted MSE（peak_quantile=0.8）
3. 訓練策略（對齊 main.py 與 training log）
    - AdamW + ReduceLROnPlateau
    - log1p target transform + expm1 inverse
    - 固定 100 epochs，無 early stopping
    - validation split（2016-10-01 之後）
4. 為什麼最終選擇 ResNet1D
    - RMSLE 優於 MLP（Train: 0.5589 vs 0.5781；Test: 0.5600 vs 0.5790）
    - Overfit Gap 僅 0.001120（Risk: No）
    - Test Peak Recall: 0.446825
5. 與題目規則嚴格對齊
    - 測試集 visitors=0 完全排除後再評估
    - RMSLE 作為主指標，同時補充 Accuracy、R2、RMSE、MAE 等輔助指標

Process timeline (what was done step-by-step):
1. 修正資料管線（日期處理、跨系統店家映射、預約聚合、reindex 補日、缺值填補）。
2. 建立 leakage-safe 特徵（lag/rolling）與 train-only encoding/scaling。
3. 建立單點預測（MLP）與序列預測（ResNet1D）並統一訓練框架。
4. 導入 AdamW + scheduler + log1p + Softplus，提升訓練穩定性。
5. 固定最終參數並鎖定交付結果。

Reproducibility command used for final delivery:
python main.py --models mlp,resnet1d --mlp-epochs 100 --resnet-epochs 100 --sequence-length 14 --mlp-lr 0.001 --resnet-lr 0.0002 --target-transform log1p --val-start-date 2016-10-01 --nn-log-interval 5

Comments on the course:
這次 take-home 讓我實際體會到時間序列任務中，資料前處理與評估定義的重要性往往不亞於模型本身。透過逐步修正資料管線、特徵工程與訓練策略，我更理解如何在可重現條件下，將模型表現與題目目標真正對齊。
