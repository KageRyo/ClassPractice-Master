from typing import List
from pydantic import BaseModel


class StoreInfoSchema(BaseModel):
    air_store_id: str
    air_genre_name: str
    air_area_name: str
    latitude: float
    longitude: float


class VisitDataSchema(BaseModel):
    air_store_id: str
    visit_date: str
    visitors: int


class ReserveDataSchema(BaseModel):
    air_store_id: str
    visit_datetime: str
    reserve_datetime: str
    reserve_visitors: int


class DateInfoSchema(BaseModel):
    calendar_date: str
    day_of_week: str
    holiday_flg: int


class ProcessedDataSchema(BaseModel):
    features: List[float]
    target: float
