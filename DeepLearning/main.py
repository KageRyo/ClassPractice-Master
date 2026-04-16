import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_loading import load_data
from src.preprocessing.preprocessing import preprocess_data
from src.models.models_training import train_xgb, train_linear, train_rf, train_mlp, train_lstm
from src.evaluation.evaluation import rmsle, evaluate_model, save_results
import os

# 設定隨機種子
np.random.seed(42)

# 資料路徑
data_path = 'datasets/'

# 載入資料
print("載入資料...")
air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info = load_data(data_path)
print("資料載入完成。")

# 預處理
data = preprocess_data(air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info)

# 基本探索
print(f"訪客人數統計:\n{data['visitors'].describe()}")

# 繪圖：訪客人數分布
plt.figure(figsize=(10, 6))
sns.histplot(data['visitors'], bins=50, kde=True)
plt.title('Visitor Count Distribution')
plt.xlabel('Visitors')
plt.ylabel('Frequency')
plt.savefig('visitors_distribution.png')
plt.show()

# 特徵和目標
features = ['reserve_visitors', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'month', 'day', 'dayofweek', 'is_holiday']
target = 'visitors'

# 分割訓練和測試 (2016 訓練, 2017 測試)
train_data = data[data['year'] == 2016]
test_data = data[data['year'] == 2017]

# 依店家與日期排序，避免時間序列跨店家串接
train_data = train_data.sort_values(['air_store_id', 'visit_date']).reset_index(drop=True)
test_data = test_data.sort_values(['air_store_id', 'visit_date']).reset_index(drop=True)

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print(f"訓練資料形狀: {X_train.shape}")
print(f"測試資料形狀: {X_test.shape}")

# 題意要求：測試集中店休日(0 visitors)不納入評分
test_open_mask = y_test > 0
X_test_open = X_test[test_open_mask]
y_test_open = y_test[test_open_mask]
print(f"測試評分樣本(排除 0 visitors): {X_test_open.shape[0]}")

# 建立 models 資料夾
os.makedirs('models', exist_ok=True)

# 訓練和評估模型
models_to_train = ['linear', 'rf', 'xgb', 'mlp', 'lstm']  # 可以修改這裡選擇模型
results = []

# 準備序列 for LSTM
sequence_length = 7
def create_sequences(data, seq_length, features, target):
    X, y = [], []

    # 逐店家建立序列，避免不同店家的時間點被接在同一個序列
    for _, store_data in data.groupby('air_store_id'):
        store_data = store_data.sort_values('visit_date').reset_index(drop=True)
        if len(store_data) <= seq_length:
            continue

        store_features = store_data[features].values
        store_targets = store_data[target].values
        for i in range(len(store_data) - seq_length):
            X.append(store_features[i:i+seq_length])
            y.append(store_targets[i+seq_length])

    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(train_data, sequence_length, features, target)
X_test_seq, y_test_seq = create_sequences(test_data, sequence_length, features, target)

for model_name in models_to_train:
    try:
        print(f"\n訓練 {model_name}...")
        if model_name == 'linear':
            model = train_linear(X_train, y_train, f'models/{model_name}_model.pkl')
            train_rmsle = evaluate_model(model, X_train, y_train, model_name)
            test_rmsle = evaluate_model(model, X_test_open, y_test_open, model_name)
        elif model_name == 'rf':
            model = train_rf(X_train, y_train, f'models/{model_name}_model.pkl')
            train_rmsle = evaluate_model(model, X_train, y_train, model_name)
            test_rmsle = evaluate_model(model, X_test_open, y_test_open, model_name)
        elif model_name == 'xgb':
            model = train_xgb(X_train, y_train, f'models/{model_name}_model.pkl')
            train_rmsle = evaluate_model(model, X_train, y_train, model_name)
            test_rmsle = evaluate_model(model, X_test_open, y_test_open, model_name)
        elif model_name == 'mlp':
            model = train_mlp(X_train, y_train, input_size=len(features), save_path=f'models/{model_name}_model.pth')
            train_rmsle = evaluate_model(model, X_train, y_train, model_name)
            test_rmsle = evaluate_model(model, X_test_open, y_test_open, model_name)
        elif model_name == 'lstm':
            model = train_lstm(X_train_seq, y_train_seq, input_size=len(features), save_path=f'models/{model_name}_model.pth')
            test_seq_open_mask = y_test_seq > 0
            X_test_seq_open = X_test_seq[test_seq_open_mask]
            y_test_seq_open = y_test_seq[test_seq_open_mask]
            train_rmsle = evaluate_model(model, X_train_seq, y_train_seq, model_name)
            test_rmsle = evaluate_model(model, X_test_seq_open, y_test_seq_open, model_name)

        print(f"{model_name} Train RMSLE: {train_rmsle}")
        print(f"{model_name} Test RMSLE: {test_rmsle}")

        results.append({
            'Model': model_name,
            'Train_RMSLE': train_rmsle,
            'Test_RMSLE': test_rmsle
        })
    except Exception as e:
        print(f"訓練 {model_name} 時發生錯誤: {e}")
        continue

save_results(results)
print("結果已儲存到 results.csv")
print("模型已儲存到 models/ 資料夾")
