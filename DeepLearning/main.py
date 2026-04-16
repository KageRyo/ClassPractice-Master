import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from loguru import logger
from src.data.data_loading import load_data
from src.preprocessing.preprocessing import preprocess_data
from src.models.models_training import train_xgb, train_linear, train_rf, train_mlp, train_lstm, train_lgbm, train_catboost
from src.evaluation.evaluation import evaluate_regression_metrics, save_results
from src.schemas.training_schema import ModelTypeSchema, OverfittingFlagSchema, TrainingResultSchema, ModelMetadataSchema
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


def parse_args():
    parser = argparse.ArgumentParser(description='Restaurant visitors training pipeline')
    parser.add_argument('--data-path', type=str, default='datasets/', help='Path to dataset directory')
    parser.add_argument('--models', type=str, default='linear,rf,xgb,lgbm,catboost,mlp,lstm', help='Comma-separated model list')
    parser.add_argument('--sequence-length', type=int, default=7, help='Sequence length for LSTM')
    parser.add_argument('--mlp-epochs', type=int, default=100, help='Training epochs for MLP')
    parser.add_argument('--lstm-epochs', type=int, default=30, help='Training epochs for LSTM')
    parser.add_argument('--mlp-hidden-size', type=int, default=64, help='Hidden size for MLP')
    parser.add_argument('--lstm-hidden-size', type=int, default=64, help='Hidden size for LSTM')
    parser.add_argument('--lstm-num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--nn-log-interval', type=int, default=5, help='Epoch interval for NN training logs')
    parser.add_argument('--overfit-gap-threshold', type=float, default=0.08, help='Threshold of (Test_RMSLE - Train_RMSLE) to flag overfitting')
    parser.add_argument('--skip-plot', action='store_true', help='Disable matplotlib plotting')
    return parser.parse_args()


args = parse_args()


def get_model_metadata(model_name, args_obj):
    if model_name == 'linear':
        return ModelMetadataSchema(
            Num_Layers=1,
            Units='9 -> 1',
            Activation='N/A (linear)',
            Loss_Function='MSE (least squares)',
            Cost_Function='MSE (least squares)',
            Epochs=1,
        )
    if model_name == 'rf':
        return ModelMetadataSchema(
            Num_Layers=0,
            Units='n_estimators=100',
            Activation='N/A (tree ensemble)',
            Loss_Function='MSE criterion',
            Cost_Function='MSE criterion',
            Epochs=1,
        )
    if model_name == 'xgb':
        return ModelMetadataSchema(
            Num_Layers=0,
            Units='n_estimators=100, max_depth=6',
            Activation='N/A (boosted trees)',
            Loss_Function='reg:squarederror (MSE)',
            Cost_Function='reg:squarederror (MSE)',
            Epochs=100,
        )
    if model_name == 'lgbm':
        return ModelMetadataSchema(
            Num_Layers=0,
            Units='n_estimators=300, num_leaves=31',
            Activation='N/A (gradient boosting tree)',
            Loss_Function='L2/RMSE objective',
            Cost_Function='L2/RMSE objective',
            Epochs=300,
        )
    if model_name == 'catboost':
        return ModelMetadataSchema(
            Num_Layers=0,
            Units='iterations=500, depth=6',
            Activation='N/A (oblivious trees)',
            Loss_Function='RMSE',
            Cost_Function='RMSE',
            Epochs=500,
        )
    if model_name == 'mlp':
        return ModelMetadataSchema(
            Num_Layers=2,
            Units=f'9 -> {args_obj.mlp_hidden_size} -> 1',
            Activation='ReLU (hidden)',
            Loss_Function='MSELoss',
            Cost_Function='MSELoss',
            Epochs=args_obj.mlp_epochs,
        )
    if model_name == 'lstm':
        return ModelMetadataSchema(
            Num_Layers=args_obj.lstm_num_layers + 1,
            Units=f'LSTM(hidden={args_obj.lstm_hidden_size}, layers={args_obj.lstm_num_layers}) -> FC(1)',
            Activation='LSTM gates + Linear output',
            Loss_Function='MSELoss',
            Cost_Function='MSELoss',
            Epochs=args_obj.lstm_epochs,
        )
    raise ValueError(f'Unsupported model for metadata: {model_name}')


def write_exam_summary(results_df, output_path='best_model_summary.md'):
    if results_df.empty:
        return

    best_row = results_df.sort_values('Test_RMSLE', ascending=True).iloc[0]
    lines = [
        '# Best Model Summary',
        '',
        f"Best Model: {best_row['Model']}",
        f"Train RMSLE: {best_row['Train_RMSLE']:.6f}",
        f"Test RMSLE: {best_row['Test_RMSLE']:.6f}",
        f"Train RMSE: {best_row['Train_RMSE']:.6f}",
        f"Test RMSE: {best_row['Test_RMSE']:.6f}",
        f"Train MAE: {best_row['Train_MAE']:.6f}",
        f"Test MAE: {best_row['Test_MAE']:.6f}",
        f"Train R2: {best_row['Train_R2']:.6f}",
        f"Test R2: {best_row['Test_R2']:.6f}",
        f"Test Peak Recall: {best_row['Test_Peak_Recall']:.6f}",
        f"Overfit Gap (Test-Train): {best_row['Overfit_Gap']:.6f}",
        f"Overfitting Risk: {best_row['Overfitting_Flag']}",
        '',
        'Model Information (for Mid-Term form):',
        f"- Number of layers: {best_row['Num_Layers']}",
        f"- Number of units in each layer: {best_row['Units']}",
        f"- Activation functions used: {best_row['Activation']}",
        f"- Loss function: {best_row['Loss_Function']}",
        f"- Cost function: {best_row['Cost_Function']}",
        f"- Training epochs: {best_row['Epochs']}",
        f"- Training RMSLE: {best_row['Train_RMSLE']:.6f}",
        f"- Testing RMSLE: {best_row['Test_RMSLE']:.6f}",
        f"- Training R2 (%): {max(0.0, best_row['Train_R2']) * 100:.2f}",
        f"- Testing R2 (%): {max(0.0, best_row['Test_R2']) * 100:.2f}",
        f"- Testing Peak Recall: {best_row['Test_Peak_Recall']:.6f}",
        '',
        'Note: This project uses RMSLE as required. If your report requires % accuracy, clarify conversion method.',
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# 設定隨機種子
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 資料路徑
data_path = args.data_path

# 載入資料
logger.info('載入資料...')
air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info = load_data(data_path)
logger.info('資料載入完成。')

# 預處理
data = preprocess_data(air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info)

# 基本探索
logger.info(f"訪客人數統計:\n{data['visitors'].describe()}")

# 繪圖：訪客人數分布
if not args.skip_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['visitors'], bins=50, kde=True)
    plt.title('Visitor Count Distribution')
    plt.xlabel('Visitors')
    plt.ylabel('Frequency')
    plt.savefig('visitors_distribution.png')
    logger.info('已輸出圖表: visitors_distribution.png')
    plt.show()
else:
    logger.info('已略過訪客分布圖輸出 (--skip-plot)')

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
allowed_models = {model.value for model in ModelTypeSchema}
models_to_train = [m.strip().lower() for m in args.models.split(',') if m.strip()]
invalid_models = [m for m in models_to_train if m not in allowed_models]
if invalid_models:
    raise ValueError(f'Unsupported models: {invalid_models}. Allowed: {sorted(allowed_models)}')

logger.info(
    f'本次訓練設定 | models={models_to_train} | mlp_epochs={args.mlp_epochs} '
    f'| lstm_epochs={args.lstm_epochs} | sequence_length={args.sequence_length}'
)

results = []


def add_prefix(metric_dict, prefix):
    return {f'{prefix}_{k}': v for k, v in metric_dict.items()}

# 準備序列 for LSTM
sequence_length = args.sequence_length
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

if 'lstm' in models_to_train:
    logger.info(f'LSTM 序列樣本數 | train={len(X_train_seq)}, test={len(X_test_seq)}')

for model_name in models_to_train:
    try:
        model_start = time.perf_counter()
        logger.info(f'訓練 {model_name}...')
        if model_name == 'linear':
            model = train_linear(X_train, y_train, f'models/{model_name}_model.pkl')
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'rf':
            model = train_rf(X_train, y_train, f'models/{model_name}_model.pkl')
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'xgb':
            model = train_xgb(X_train, y_train, f'models/{model_name}_model.pkl')
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'lgbm':
            model = train_lgbm(X_train, y_train, f'models/{model_name}_model.pkl')
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'catboost':
            model = train_catboost(X_train, y_train, f'models/{model_name}_model.pkl')
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'mlp':
            model = train_mlp(
                X_train,
                y_train,
                input_size=len(features),
                hidden_size=args.mlp_hidden_size,
                num_epochs=args.mlp_epochs,
                log_interval=args.nn_log_interval,
                progress_callback=lambda msg: logger.info(msg),
                save_path=f'models/{model_name}_model.pth'
            )
            train_metrics = evaluate_regression_metrics(model, X_train, y_train, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_open, y_test_open, model_name)
        elif model_name == 'lstm':
            model = train_lstm(
                X_train_seq,
                y_train_seq,
                input_size=len(features),
                hidden_size=args.lstm_hidden_size,
                num_layers=args.lstm_num_layers,
                num_epochs=args.lstm_epochs,
                log_interval=args.nn_log_interval,
                progress_callback=lambda msg: logger.info(msg),
                save_path=f'models/{model_name}_model.pth'
            )
            test_seq_open_mask = y_test_seq > 0
            X_test_seq_open = X_test_seq[test_seq_open_mask]
            y_test_seq_open = y_test_seq[test_seq_open_mask]
            train_metrics = evaluate_regression_metrics(model, X_train_seq, y_train_seq, model_name)
            test_metrics = evaluate_regression_metrics(model, X_test_seq_open, y_test_seq_open, model_name)

        logger.info(
            f"{model_name} Train | RMSLE={train_metrics['RMSLE']:.6f} RMSE={train_metrics['RMSE']:.6f} "
            f"MAE={train_metrics['MAE']:.6f} R2={train_metrics['R2']:.6f} PeakRecall={train_metrics['Peak_Recall']:.6f}"
        )
        logger.info(
            f"{model_name} Test  | RMSLE={test_metrics['RMSLE']:.6f} RMSE={test_metrics['RMSE']:.6f} "
            f"MAE={test_metrics['MAE']:.6f} R2={test_metrics['R2']:.6f} PeakRecall={test_metrics['Peak_Recall']:.6f}"
        )
        elapsed = time.perf_counter() - model_start
        logger.info(f'{model_name} 訓練耗時: {elapsed:.2f}s')

        meta = get_model_metadata(model_name, args)
        overfit_gap = float(test_metrics['RMSLE'] - train_metrics['RMSLE'])
        overfit_flag = OverfittingFlagSchema.YES.value if overfit_gap > args.overfit_gap_threshold else OverfittingFlagSchema.NO.value

        result_payload = {
            'Model': model_name,
            **add_prefix(train_metrics, 'Train'),
            **add_prefix(test_metrics, 'Test'),
            'Overfit_Gap': overfit_gap,
            'Overfitting_Flag': overfit_flag,
            'Train_Time_Seconds': round(elapsed, 2),
            **meta.model_dump(),
        }
        validated_result = TrainingResultSchema(**result_payload)
        results.append(validated_result.model_dump())

        logger.info(f'{model_name} Overfit Gap: {overfit_gap:.6f} (threshold={args.overfit_gap_threshold}) | Overfitting={overfit_flag}')
    except Exception as e:
        logger.exception(f'訓練 {model_name} 時發生錯誤: {e}')
        continue

save_results(results)
logger.info('結果已儲存到 results.csv')
logger.info('模型已儲存到 models/ 資料夾')

results_df = pd.DataFrame(results)
if not results_df.empty:
    best_model_row = results_df.sort_values('Test_RMSLE', ascending=True).iloc[0]
    logger.info(
        f"最佳模型: {best_model_row['Model']} | Train RMSLE={best_model_row['Train_RMSLE']:.6f} "
        f"| Test RMSLE={best_model_row['Test_RMSLE']:.6f} | Overfitting={best_model_row['Overfitting_Flag']}"
    )
    write_exam_summary(results_df)
    logger.info('已輸出最佳模型摘要: best_model_summary.md')
