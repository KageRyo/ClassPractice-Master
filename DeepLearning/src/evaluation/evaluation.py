import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_log_error, r2_score
import pandas as pd
import joblib

def rmsle(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true), 0)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))


def predict_model(model, X_test, model_type='xgb'):
    if model_type in ['xgb', 'linear', 'rf', 'lgbm', 'catboost']:
        return np.asarray(model.predict(X_test))

    if model_type in ['mlp', 'lstm', 'cnn1d', 'resnet1d', 'transformer']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        if model_type == 'mlp':
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test)))
        else:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test)))

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                y_pred.extend(np.atleast_1d(np.maximum(preds.squeeze().cpu().numpy(), 0)))
        return np.asarray(y_pred)

    raise ValueError(f'Unsupported model_type: {model_type}')


def evaluate_regression_metrics(model, X_test, y_test, model_type='xgb'):
    y_true = np.asarray(y_test, dtype=float)
    y_pred = predict_model(model, X_test, model_type=model_type)

    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    # Guard against division by zero in percentage metrics.
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100)
    mspe = float(np.nanmean((err / denom) ** 2) * 100)

    metrics = {
        'SSE': sse,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': mape,
        'MSPE': mspe,
        'RMSLE': float(rmsle(y_true, y_pred)),
    }

    # Optional regression-oriented "recall": recall of high-demand days (top 20%).
    peak_threshold = float(np.quantile(y_true, 0.8))
    true_peak = y_true >= peak_threshold
    pred_peak = y_pred >= peak_threshold
    tp = int(np.sum(true_peak & pred_peak))
    fn = int(np.sum(true_peak & (~pred_peak)))
    peak_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['Peak_Recall'] = peak_recall

    return metrics

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    y_pred = predict_model(model, X_test, model_type=model_type)
    return rmsle(y_test, y_pred)

def load_model(model_type, path):
    if model_type in ['xgb', 'linear', 'rf', 'lgbm', 'catboost']:
        return joblib.load(path)
    elif model_type == 'mlp':
        from src.models.models_training import MLPModel
        model = MLPModel(input_size=14, hidden_size=64, output_size=1)
        model.load_state_dict(torch.load(path))
        return model
    elif model_type == 'lstm':
        from src.models.models_training import OptimizedLSTMModel
        model = OptimizedLSTMModel(input_size=14, hidden_size=64, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(path))
        return model
    elif model_type == 'cnn1d':
        from src.models.models_training import CNN1DModel
        model = CNN1DModel(input_size=14, sequence_length=7, output_size=1)
        model.load_state_dict(torch.load(path))
        return model
    elif model_type == 'resnet1d':
        from src.models.models_training import ResNet1DModel
        model = ResNet1DModel(input_size=14, output_size=1)
        model.load_state_dict(torch.load(path))
        return model
    elif model_type == 'transformer':
        from src.models.models_training import TimeSeriesTransformerModel
        model = TimeSeriesTransformerModel(input_size=14, d_model=64, nhead=4, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(path))
        return model

def save_results(results, filename='results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
