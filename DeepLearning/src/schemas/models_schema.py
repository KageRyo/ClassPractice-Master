from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class ModelType(Enum):
    LINEAR = "linear"
    RF = "rf"
    XGB = "xgb"
    MLP = "mlp"
    LSTM = "lstm"

class StoreInfo(BaseModel):
    air_store_id: str
    air_genre_name: str
    air_area_name: str
    latitude: float
    longitude: float

class VisitData(BaseModel):
    air_store_id: str
    visit_date: str
    visitors: int

class ReserveData(BaseModel):
    air_store_id: str
    visit_datetime: str
    reserve_datetime: str
    reserve_visitors: int

class DateInfo(BaseModel):
    calendar_date: str
    day_of_week: str
    holiday_flg: int

class ProcessedData(BaseModel):
    features: List[float]
    target: float
