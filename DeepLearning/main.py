import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from src.data.data_loading import load_data
from src.preprocessing.preprocessing import preprocess_data
from src.models.models_training import train_xgb, train_linear, train_rf, train_mlp, train_lstm
from src.evaluation.evaluation import rmsle, evaluate_model, save_results
import os
import sys


def setup_logger():
    os.makedirs('logs', exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level='INFO',
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>'
    )
    logger.add(
        'logs/training.log',
        level='INFO',
        rotation='1 MB',
        encoding='utf-8',
        format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'
    )


setup_logger()

# 設定隨機種子
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 資料路徑
data_path = 'datasets/'

# 載入資料
logger.info('載入資料...')
air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info = load_data(data_path)
logger.info('資料載入完成。')

# 預處理
data = preprocess_data(air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info)

# 基本探索
logger.info(f"訪客人數統計:\n{data['visitors'].describe()}")

# 繪圖：訪客人數分布
plt.figure(figsize=(10, 6))
sns.histplot(data['visitors'], bins=50, kde=True)
plt.title('Visitor Count Distribution')
plt.xlabel('Visitors')
plt.ylabel('Frequency')
plt.savefig('visitors_distribution.png')
logger.info('已輸出圖表: visitors_distribution.png')
plt.show()

# 特徵和目標
features = ['reserve_visitors', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'month', 'day', 'dayofweek', 'is_holiday']
target = 'visitors'

# 分割訓練和測試 (2016 訓練, 2017 測試)
train_data = data[data['year'] == 2016].copy()
test_data = data[data['year'] == 2017].copy()

# 依店家與日期排序，避免時間序列跨店家串接
train_data = train_data.sort_values(['air_store_id', 'visit_date']).reset_index(drop=True)
test_data = test_data.sort_values(['air_store_id', 'visit_date']).reset_index(drop=True)

def encode_with_train_mapping(train_df, test_df, col_name):
    train_values = train_df[col_name].astype(str)
    test_values = test_df[col_name].astype(str)
    mapping = {value: idx for idx, value in enumerate(sorted(train_values.unique()))}
    train_df[col_name] = train_values.map(mapping).astype(int)
    test_df[col_name] = test_values.map(mapping).fillna(-1).astype(int)

# 只使用訓練集建立類別編碼，避免資料洩漏
encode_with_train_mapping(train_data, test_data, 'air_genre_name')
encode_with_train_mapping(train_data, test_data, 'air_area_name')

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

logger.info(f'訓練資料形狀: {X_train.shape}')
logger.info(f'測試資料形狀: {X_test.shape}')

# 題意要求：測試集中店休日(0 visitors)不納入評分
test_open_mask = y_test > 0
X_test_open = X_test[test_open_mask]
y_test_open = y_test[test_open_mask]
logger.info(f'測試評分樣本(排除 0 visitors): {X_test_open.shape[0]}')

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
        logger.info(f'訓練 {model_name}...')
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

        logger.info(f'{model_name} Train RMSLE: {train_rmsle}')
        logger.info(f'{model_name} Test RMSLE: {test_rmsle}')

        results.append({
            'Model': model_name,
            'Train_RMSLE': train_rmsle,
            'Test_RMSLE': test_rmsle
        })
    except Exception as e:
        logger.exception(f'訓練 {model_name} 時發生錯誤: {e}')
        continue

save_results(results)
logger.info('結果已儲存到 results.csv')
logger.info('模型已儲存到 models/ 資料夾')
