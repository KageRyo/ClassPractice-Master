import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info):
    # 日期處理
    air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
    air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
    air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
    hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
    hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
    date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])

    # 合併 hpg 和 air 預約
    hpg_reserve = hpg_reserve.merge(store_relation, on='hpg_store_id', how='left')
    hpg_reserve = hpg_reserve.dropna(subset=['air_store_id'])

    # 合併所有預約
    all_reserve = pd.concat([air_reserve[['air_store_id', 'visit_datetime', 'reserve_visitors']],
                             hpg_reserve[['air_store_id', 'visit_datetime', 'reserve_visitors']]], ignore_index=True)

    # 按日期和商店聚合預約
    all_reserve['visit_date'] = all_reserve['visit_datetime'].dt.date
    all_reserve['visit_date'] = pd.to_datetime(all_reserve['visit_date'])
    reserve_agg = all_reserve.groupby(['air_store_id', 'visit_date'])['reserve_visitors'].sum().reset_index()

    # 合併訪客資料和預約資料
    observed = air_visit.merge(reserve_agg, on=['air_store_id', 'visit_date'], how='left')
    observed['reserve_visitors'] = observed['reserve_visitors'].fillna(0)

    # 使用全日期範圍補齊每家店每日紀錄，確保時間軸連續
    full_dates = pd.date_range(date_info['calendar_date'].min(), date_info['calendar_date'].max(), freq='D')
    full_index = pd.MultiIndex.from_product(
        [air_store['air_store_id'].unique(), full_dates],
        names=['air_store_id', 'visit_date']
    )
    full_data = pd.DataFrame(index=full_index).reset_index()

    data = full_data.merge(observed, on=['air_store_id', 'visit_date'], how='left')

    # 合併商店資訊與日期資訊
    data = data.merge(air_store, on='air_store_id', how='left')
    data = data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')

    # 缺值補齊
    data['visitors'] = data['visitors'].fillna(0)
    data['reserve_visitors'] = data['reserve_visitors'].fillna(0)
    data['holiday_flg'] = data['holiday_flg'].fillna(0)
    data['day_of_week'] = data['day_of_week'].fillna(data['visit_date'].dt.day_name())
    data['air_genre_name'] = data['air_genre_name'].fillna('Unknown')
    data['air_area_name'] = data['air_area_name'].fillna('Unknown')

    # 特徵工程
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    data['day'] = data['visit_date'].dt.day
    data['dayofweek'] = data['visit_date'].dt.dayofweek
    data['is_holiday'] = data['holiday_flg']

    numeric_fill_cols = ['latitude', 'longitude', 'reserve_visitors', 'month', 'day', 'dayofweek']
    for col in numeric_fill_cols:
        data[col] = data[col].fillna(0)

    return data.sort_values(['air_store_id', 'visit_date']).reset_index(drop=True)


def create_sliding_window_sequences(data, seq_length, features, target):
    X, y, target_dates, store_ids = [], [], [], []

    for store_id, store_data in data.groupby('air_store_id'):
        store_data = store_data.sort_values('visit_date').reset_index(drop=True)
        if len(store_data) <= seq_length:
            continue

        store_features = store_data[features].values
        store_targets = store_data[target].values
        store_dates = store_data['visit_date'].values

        for i in range(seq_length, len(store_data)):
            X.append(store_features[i - seq_length:i])
            y.append(store_targets[i])
            target_dates.append(store_dates[i])
            store_ids.append(store_id)

    return np.asarray(X), np.asarray(y), np.asarray(target_dates), np.asarray(store_ids)


def split_sequences_by_target_year(X, y, target_dates, train_years=(2016,), test_year=2017):
    target_dates = pd.to_datetime(target_dates)
    train_mask = target_dates.year.isin(list(train_years))
    test_mask = target_dates.year == test_year

    return {
        'X_train_seq': X[train_mask],
        'y_train_seq': y[train_mask],
        'X_test_seq': X[test_mask],
        'y_test_seq': y[test_mask],
        'train_target_dates': target_dates[train_mask],
        'test_target_dates': target_dates[test_mask],
    }


def fit_standard_scaler(train_df, test_df, continuous_cols):
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler.fit(train_df[continuous_cols])
    train_df[continuous_cols] = scaler.transform(train_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    return train_df, test_df, scaler


def transform_sequences_with_scaler(X_seq, features, continuous_cols, scaler):
    X_scaled = np.asarray(X_seq, dtype=float).copy()
    if X_scaled.size == 0:
        return X_scaled

    continuous_indices = [features.index(col) for col in continuous_cols]
    reshaped = X_scaled.reshape(-1, len(features))
    reshaped_cont = reshaped[:, continuous_indices]
    reshaped[:, continuous_indices] = scaler.transform(reshaped_cont)

    return reshaped.reshape(X_scaled.shape)
