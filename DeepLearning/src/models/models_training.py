import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return F.relu(out)


class OptimizedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return F.relu(out)


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.relu(out)

def train_xgb(X_train, y_train, save_path='models/xgb_model.pkl'):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model


def train_lgbm(X_train, y_train, save_path='models/lgbm_model.pkl'):
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model


def train_catboost(X_train, y_train, save_path='models/catboost_model.pkl'):
    model = CatBoostRegressor(
        loss_function='RMSE',
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def train_linear(X_train, y_train, save_path='models/linear_model.pkl'):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def train_rf(X_train, y_train, save_path='models/rf_model.pkl'):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def train_mlp(
    X_train,
    y_train,
    input_size,
    hidden_size=64,
    num_epochs=100,
    log_interval=5,
    progress_callback=None,
    save_path='models/mlp_model.pth'
):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MLPModel(input_size, hidden_size, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(train_dataset)
        scheduler.step(avg_loss)
        if progress_callback and ((epoch + 1) == 1 or (epoch + 1) == num_epochs or (epoch + 1) % max(log_interval, 1) == 0):
            lr = optimizer.param_groups[0]['lr']
            progress_callback(f"MLP Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} - LR: {lr:.6e}")

    torch.save(model.state_dict(), save_path)
    return model

def train_lstm(
    X_train_seq,
    y_train_seq,
    input_size,
    hidden_size=64,
    num_layers=2,
    num_epochs=30,
    log_interval=5,
    progress_callback=None,
    save_path='models/lstm_model.pth'
):
    return train_optimized_lstm(
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_epochs=num_epochs,
        log_interval=log_interval,
        progress_callback=progress_callback,
        save_path=save_path,
    )


def _train_sequence_regressor(
    model,
    X_train_seq,
    y_train_seq,
    num_epochs=30,
    log_interval=5,
    progress_callback=None,
    save_path='models/sequence_model.pth',
    model_name='SequenceModel',
):
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(train_dataset)
        scheduler.step(avg_loss)
        if progress_callback and ((epoch + 1) == 1 or (epoch + 1) == num_epochs or (epoch + 1) % max(log_interval, 1) == 0):
            lr = optimizer.param_groups[0]['lr']
            progress_callback(
                f"{model_name} Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} - LR: {lr:.6e}"
            )

    torch.save(model.state_dict(), save_path)
    return model


def train_optimized_lstm(
    X_train_seq,
    y_train_seq,
    input_size,
    hidden_size=128,
    num_layers=2,
    num_epochs=40,
    log_interval=5,
    progress_callback=None,
    save_path='models/lstm_model.pth',
):
    model = OptimizedLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=0.3,
    )
    return _train_sequence_regressor(
        model=model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        num_epochs=num_epochs,
        log_interval=log_interval,
        progress_callback=progress_callback,
        save_path=save_path,
        model_name='OptimizedLSTM',
    )
