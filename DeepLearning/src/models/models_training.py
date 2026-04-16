import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import math
import numpy as np
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


class CNN1DModel(nn.Module):
    def __init__(self, input_size, sequence_length, output_size=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        pooled_len = max(1, sequence_length // 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * pooled_len, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.relu(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet1DModel(nn.Module):
    def __init__(self, input_size, output_size=1, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock1D(64, 64, kernel_size=3, dropout=dropout)
        self.block2 = ResidualBlock1D(64, 128, kernel_size=3, dropout=dropout)
        self.block3 = ResidualBlock1D(128, 128, kernel_size=3, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.head(x)
        return F.softplus(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3, output_size=1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=256,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.relu(x)


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
        return F.softplus(out)

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
    save_path='models/mlp_model.pth',
    X_val=None,
    y_val=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
    lr=1e-3,
):
    y_train_arr = y_train.values.astype(np.float32)
    if target_transform == 'log1p':
        y_train_arr = np.log1p(np.clip(y_train_arr, a_min=0, a_max=None))

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        y_val_arr = y_val.values.astype(np.float32)
        if target_transform == 'log1p':
            y_val_arr = np.log1p(np.clip(y_val_arr, a_min=0, a_max=None))
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_arr, dtype=torch.float32)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel(input_size, hidden_size, 1)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_state = None
    best_score = float('inf')
    stale_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(train_dataset)

        monitor_loss = avg_loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch = X_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    preds = model(X_val_batch)
                    batch_loss = criterion(preds.squeeze(), y_val_batch)
                    val_running += batch_loss.item() * len(X_val_batch)
            val_loss = val_running / len(val_loader.dataset)
            monitor_loss = val_loss

        scheduler.step(monitor_loss)

        if monitor_loss + min_delta < best_score:
            best_score = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if progress_callback and ((epoch + 1) == 1 or (epoch + 1) == num_epochs or (epoch + 1) % max(log_interval, 1) == 0):
            lr = optimizer.param_groups[0]['lr']
            if val_loss is None:
                progress_callback(f"MLP Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} - LR: {lr:.6e}")
            else:
                progress_callback(
                    f"MLP Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} "
                    f"- Val MSE Loss: {val_loss:.6f} - LR: {lr:.6e}"
                )

        if (
            early_stopping_enabled
            and val_loader is not None
            and (epoch + 1) >= max(min_epochs_before_stop, 1)
            and stale_epochs >= patience
        ):
            if progress_callback:
                progress_callback(f"MLP Early stopping at epoch {epoch + 1} (best monitor loss={best_score:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.target_transform = target_transform

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
    save_path='models/lstm_model.pth',
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
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
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        patience=patience,
        min_delta=min_delta,
        target_transform=target_transform,
        min_epochs_before_stop=min_epochs_before_stop,
        early_stopping_enabled=early_stopping_enabled,
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
    lr=1e-3,
    weight_decay=1e-4,
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
):
    y_train_arr = np.asarray(y_train_seq, dtype=np.float32)
    if target_transform == 'log1p':
        y_train_arr = np.log1p(np.clip(y_train_arr, a_min=0, a_max=None))

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = None
    if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
        y_val_arr = np.asarray(y_val_seq, dtype=np.float32)
        if target_transform == 'log1p':
            y_val_arr = np.log1p(np.clip(y_val_arr, a_min=0, a_max=None))
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_arr, dtype=torch.float32)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_state = None
    best_score = float('inf')
    stale_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(train_dataset)

        monitor_loss = avg_loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch = X_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    preds = model(X_val_batch)
                    batch_loss = criterion(preds.squeeze(), y_val_batch)
                    val_running += batch_loss.item() * len(X_val_batch)
            val_loss = val_running / len(val_loader.dataset)
            monitor_loss = val_loss

        scheduler.step(monitor_loss)

        if monitor_loss + min_delta < best_score:
            best_score = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if progress_callback and ((epoch + 1) == 1 or (epoch + 1) == num_epochs or (epoch + 1) % max(log_interval, 1) == 0):
            lr = optimizer.param_groups[0]['lr']
            if val_loss is None:
                progress_callback(
                    f"{model_name} Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} - LR: {lr:.6e}"
                )
            else:
                progress_callback(
                    f"{model_name} Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {avg_loss:.6f} "
                    f"- Val MSE Loss: {val_loss:.6f} - LR: {lr:.6e}"
                )

        if (
            early_stopping_enabled
            and val_loader is not None
            and (epoch + 1) >= max(min_epochs_before_stop, 1)
            and stale_epochs >= patience
        ):
            if progress_callback:
                progress_callback(f"{model_name} Early stopping at epoch {epoch + 1} (best monitor loss={best_score:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.target_transform = target_transform

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
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
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
        lr=1e-3,
        weight_decay=1e-4,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        patience=patience,
        min_delta=min_delta,
        target_transform=target_transform,
        min_epochs_before_stop=min_epochs_before_stop,
        early_stopping_enabled=early_stopping_enabled,
    )


def train_cnn1d(
    X_train_seq,
    y_train_seq,
    input_size,
    sequence_length,
    num_epochs=40,
    log_interval=5,
    progress_callback=None,
    save_path='models/cnn1d_model.pth',
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
):
    model = CNN1DModel(
        input_size=input_size,
        sequence_length=sequence_length,
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
        model_name='CNN1D',
        lr=1e-3,
        weight_decay=1e-4,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        patience=patience,
        min_delta=min_delta,
        target_transform=target_transform,
        min_epochs_before_stop=min_epochs_before_stop,
        early_stopping_enabled=early_stopping_enabled,
    )


def train_transformer(
    X_train_seq,
    y_train_seq,
    input_size,
    num_epochs=40,
    log_interval=5,
    progress_callback=None,
    save_path='models/transformer_model.pth',
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
):
    model = TimeSeriesTransformerModel(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        output_size=1,
    )
    return _train_sequence_regressor(
        model=model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        num_epochs=num_epochs,
        log_interval=log_interval,
        progress_callback=progress_callback,
        save_path=save_path,
        model_name='Transformer',
        lr=1e-3,
        weight_decay=1e-4,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        patience=patience,
        min_delta=min_delta,
        target_transform=target_transform,
        min_epochs_before_stop=min_epochs_before_stop,
        early_stopping_enabled=early_stopping_enabled,
    )


def train_resnet1d(
    X_train_seq,
    y_train_seq,
    input_size,
    num_epochs=40,
    log_interval=5,
    progress_callback=None,
    save_path='models/resnet1d_model.pth',
    X_val_seq=None,
    y_val_seq=None,
    patience=12,
    min_delta=1e-4,
    target_transform='none',
    min_epochs_before_stop=0,
    early_stopping_enabled=True,
    lr=3e-4,
):
    model = ResNet1DModel(
        input_size=input_size,
        output_size=1,
        dropout=0.1,
    )
    return _train_sequence_regressor(
        model=model,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        num_epochs=num_epochs,
        log_interval=log_interval,
        progress_callback=progress_callback,
        save_path=save_path,
        model_name='ResNet1D',
        lr=lr,
        weight_decay=1e-4,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        patience=patience,
        min_delta=min_delta,
        target_transform=target_transform,
        min_epochs_before_stop=min_epochs_before_stop,
        early_stopping_enabled=early_stopping_enabled,
    )
