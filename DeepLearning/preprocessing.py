import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    data = air_visit.merge(reserve_agg, on=['air_store_id', 'visit_date'], how='left')
    data['reserve_visitors'] = data['reserve_visitors'].fillna(0)

    # 合併商店資訊
    data = data.merge(air_store, on='air_store_id', how='left')

    # 合併日期資訊
    data = data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')

    # 特徵工程
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    data['day'] = data['visit_date'].dt.day
    data['dayofweek'] = data['visit_date'].dt.dayofweek
    data['is_holiday'] = data['holiday_flg']

    # 編碼類別特徵
    le_genre = LabelEncoder()
    le_area = LabelEncoder()
    data['air_genre_name'] = le_genre.fit_transform(data['air_genre_name'])
    data['air_area_name'] = le_area.fit_transform(data['air_area_name'])

    return data
