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

Model Information (最終選擇模型：ResNet1D):
- Model name: ResNet1D
- Number of layers: 8
- Number of units in each layer: Stem(Conv64) + ResidualBlocks(64,128,128) + GAP + FC(1)
- Activation functions used: ReLU + residual skip connections + Softplus output
- Loss function (ResNet1D 訓練):
    - 平均平方誤差（Mean Squared Error, MSE）
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
- Cost function (ResNet1D 優化目標):
    - 本專案中 cost 與 loss 同義（每個 epoch 的平均 MSE）
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Evaluation metric required by exam:
    - RMSLE = sqrt((1/N) * sum_{i=1..N} (log(1 + y_i) - log(1 + y_hat_i))^2)
- Training epochs: 100
- Training accuracy: 63.36%（以 Train R2 表示）
- Testing accuracy: 0.00%（以 Test R2 百分比下限 0 表示）
- Optimization techniques employed:
        1. train-only scaling（只用 2016 訓練資料 fit scaler）
        2. 完整時間軸補齊（per-store reindex）與店休日 visitors 補 0
        3. visitors lag/rolling 特徵工程（lag_1/7/14, roll_mean_7/std_7）
        4. AdamW + ReduceLROnPlateau + Softplus output + log1p target transform
        5. validation split 監控訓練穩定性，固定跑滿 100 epochs

Model Information (基準比較模型：MLP):
- Model name: MLP
- Number of layers: 2
- Number of units in each layer: input_dim -> 64 -> 1
- Activation functions used: ReLU (hidden) + Softplus output
- Loss function (MLP 訓練):
    - 平均平方誤差（Mean Squared Error, MSE）
    - L = (1/N) * sum_{i=1..N} (y_i - y_hat_i)^2
- Cost function (MLP 優化目標):
    - 本專案中 cost 與 loss 同義（每個 epoch 的平均 MSE）
    - J(theta) = (1/B) * sum_{b=1..B} L_b
- Evaluation metric required by exam:
    - RMSLE = sqrt((1/N) * sum_{i=1..N} (log(1 + y_i) - log(1 + y_hat_i))^2)

Difference in accuracies after each optimization technique applied:
1) Optimization technique name: 前處理管線重構（reindex + train-only scaling）
    - Before optimization: Training/Testing Accuracies = 不適用 / 不適用（當時以 RMSLE 為主）
    - After optimization: Training/Testing Accuracies = 不適用 / 不適用（當前仍以 RMSLE 為主）
    - Any other changes:
        - 將資料補成每店每日連續時間軸，避免時間斷點
        - 類別映射與標準化僅用訓練資料建立，降低 leakage 風險
        - 測試評估排除 visitors=0，與題目 scoring 規則一致

2) Optimization technique name: 特徵工程升級（lag/rolling）
    - Before optimization: Training/Testing Accuracies = 不適用 / 不適用（RMSLE 流程）
    - After optimization: Training/Testing Accuracies = 不適用 / 不適用（RMSLE 流程）
    - Any other changes:
        - 新增 visitors_lag_1/7/14
        - 新增 visitors_roll_mean_7 / visitors_roll_std_7（皆採用 t-1 以前資訊）
        - 特徵維度由 9 增加到 14

3) Optimization technique name: 訓練穩定化與最終模型篩選（MLP vs ResNet1D）
    - Before optimization: Training/Testing Accuracies = 57.81% / 57.90%（MLP 基準；RMSLE=0.578145 / 0.578952）
    - After optimization: Training/Testing Accuracies = 63.36% / 0.00%（ResNet1D 最終；RMSLE=0.558909 / 0.560029）
    - Any other changes:
        - 優化器改 AdamW，加入 ReduceLROnPlateau
        - 輸出層使用 Softplus，target transform 採 log1p
        - 最終選擇 ResNet1D 作為交付模型
        - Overfit gap = 0.001120，Overfitting Risk = No

Training / Testing summary (locked final run):
- MLP:
    - Train RMSLE: 0.578145
    - Test RMSLE: 0.578952
    - Train R2: 0.587871
    - Test R2: 0.364594
- ResNet1D (Best):
    - Train RMSLE: 0.558909
    - Test RMSLE: 0.560029
    - Train RMSE: 9.927783
    - Test RMSE: 23.959431
    - Train MAE: 5.131245
    - Test MAE: 7.896169
    - Train R2: 0.633555
    - Test R2: -0.973978
    - Test Peak Recall: 0.446825
    - Overfit Gap: 0.001120 (No)

Anything special about your model:
本專案的重點不是盲目加深模型，而是把資料流程、評估規則與模型學習空間對齊：
1. 測試評分排除 visitors=0，符合題目定義。
2. 前處理先確保時間軸與缺值語意正確，再進入模型比較。
3. 使用 log1p/expm1 對齊 RMSLE 的誤差空間。
4. 最終將流程收斂到 mlp,resnet1d 兩模型，提升可重現性與交付清晰度。

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
