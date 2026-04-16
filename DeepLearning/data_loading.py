import pandas as pd
from models import StoreInfo, VisitData, ReserveData, DateInfo

def load_data(data_path: str):
    air_visit = pd.read_csv(data_path + 'air_visit_data.csv')
    air_reserve = pd.read_csv(data_path + 'air_reserve.csv')
    hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv')
    air_store = pd.read_csv(data_path + 'air_store_info.csv')
    hpg_store = pd.read_csv(data_path + 'hpg_store_info.csv')
    store_relation = pd.read_csv(data_path + 'store_id_relation.csv')
    date_info = pd.read_csv(data_path + 'date_info.csv')
    return air_visit, air_reserve, hpg_reserve, air_store, hpg_store, store_relation, date_info
