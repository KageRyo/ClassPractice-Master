from enum import Enum
from pydantic import BaseModel


class ModelTypeSchema(Enum):
    MLP = "mlp"
    RESNET1D = "resnet1d"


class OverfittingFlagSchema(Enum):
    YES = "Yes"
    NO = "No"


class ModelMetadataSchema(BaseModel):
    Num_Layers: int
    Units: str
    Activation: str
    Loss_Function: str
    Cost_Function: str
    Epochs: int


class TrainingResultSchema(BaseModel):
    Model: str
    Train_SSE: float
    Train_MSE: float
    Train_RMSE: float
    Train_MAE: float
    Train_Accuracy: float
    Train_R2: float
    Train_MAPE: float
    Train_MSPE: float
    Train_RMSLE: float
    Train_Peak_Recall: float
    Test_SSE: float
    Test_MSE: float
    Test_RMSE: float
    Test_MAE: float
    Test_Accuracy: float
    Test_R2: float
    Test_MAPE: float
    Test_MSPE: float
    Test_RMSLE: float
    Test_Peak_Recall: float
    Overfit_Gap: float
    Overfitting_Flag: str
    Train_Time_Seconds: float
    Num_Layers: int | str
    Units: str
    Activation: str
    Loss_Function: str
    Cost_Function: str
    Epochs: int | str
