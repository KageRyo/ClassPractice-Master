import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import joblib

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    if model_type in ['xgb', 'linear', 'rf']:
        y_pred = model.predict(X_test)
    elif model_type in ['mlp', 'lstm']:
        model.eval()
        if model_type == 'mlp':
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test)))
        else:  # lstm
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test)))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                preds = model(X_batch)
                y_pred.extend(preds.squeeze().numpy())
        y_pred = np.array(y_pred)
    return rmsle(y_test, y_pred)

def load_model(model_type, path):
    if model_type in ['xgb', 'linear', 'rf']:
        return joblib.load(path)
    elif model_type == 'mlp':
        from src.models.models_training import MLPModel
        model = MLPModel(input_size=9, hidden_size=64, output_size=1)
        model.load_state_dict(torch.load(path))
        return model
    elif model_type == 'lstm':
        from src.models.models_training import LSTMModel
        model = LSTMModel(input_size=9, hidden_size=64, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(path))
        return model

def save_results(results, filename='results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
