import numpy as np
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    if model_type == 'xgb':
        y_pred = model.predict(X_test)
    elif model_type == 'lstm':
        model.eval()
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

def save_results(results, filename='results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
