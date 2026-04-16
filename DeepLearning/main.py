import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_loading import load_data
from src.preprocessing.preprocessing import preprocess_data
from src.models.models_training import train_xgb, train_linear, train_rf, train_mlp
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

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print(f"訓練資料形狀: {X_train.shape}")
print(f"測試資料形狀: {X_test.shape}")

# 建立 models 資料夾
os.makedirs('models', exist_ok=True)

# 訓練和評估模型
models_to_train = ['linear', 'rf', 'xgb', 'mlp', 'lstm']  # 可以修改這裡選擇模型
results = []

# 準備序列 for LSTM
sequence_length = 7
def create_sequences(data, seq_length, features, target):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][features].values)
        y.append(data.iloc[i+seq_length][target])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(train_data, sequence_length, features, target)
X_test_seq, y_test_seq = create_sequences(test_data, sequence_length, features, target)

for model_name in models_to_train:
    try:
        print(f"\n訓練 {model_name}...")
        if model_name == 'linear':
            model = train_linear(X_train, y_train, f'models/{model_name}_model.pkl')
            rmsle_score = evaluate_model(model, X_test, y_test, model_name)
        elif model_name == 'rf':
            model = train_rf(X_train, y_train, f'models/{model_name}_model.pkl')
            rmsle_score = evaluate_model(model, X_test, y_test, model_name)
        elif model_name == 'xgb':
            model = train_xgb(X_train, y_train, f'models/{model_name}_model.pkl')
            rmsle_score = evaluate_model(model, X_test, y_test, model_name)
        elif model_name == 'mlp':
            model = train_mlp(X_train, y_train, input_size=len(features), save_path=f'models/{model_name}_model.pth')
            rmsle_score = evaluate_model(model, X_test, y_test, model_name)
        elif model_name == 'lstm':
            model = train_lstm(X_train_seq, y_train_seq, input_size=len(features), save_path=f'models/{model_name}_model.pth')
            rmsle_score = evaluate_model(model, X_test_seq, y_test_seq, model_name)

        print(f"{model_name} RMSLE: {rmsle_score}")

        # 簡化準確率
        if model_name == 'lstm':
            mean_y = np.mean(y_test_seq)
        else:
            mean_y = np.mean(y_test)
        accuracy = 1 - rmsle_score / mean_y
        results.append({
            'Model': model_name,
            'RMSLE': rmsle_score,
            'Training_Accuracy': 1 - rmsle_score / np.mean(y_train if model_name != 'lstm' else y_train_seq),
            'Testing_Accuracy': accuracy
        })
    except Exception as e:
        print(f"訓練 {model_name} 時發生錯誤: {e}")
        continue

save_results(results)
print("結果已儲存到 results.csv")
print("模型已儲存到 models/ 資料夾")
