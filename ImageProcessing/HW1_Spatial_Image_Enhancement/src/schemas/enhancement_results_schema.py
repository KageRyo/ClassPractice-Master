import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class EnhancementResultsSchema(BaseModel):
    """Validated enhancement result container (Pydantic v2).

    Validation:
    - Each field: 2D, non-empty, uint8 ndarray
    - All shapes identical
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    power_law: np.ndarray = Field(..., description="Gamma corrected image array")
    hist_eq: np.ndarray = Field(..., description="Histogram equalized image array")
    laplacian: np.ndarray = Field(..., description="Laplacian sharpened image array")

    @field_validator('power_law', 'hist_eq', 'laplacian', mode='before')
    @classmethod
    def _validate_array(cls, v, info):  # noqa: D401
        field_name = info.field_name
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{field_name} must be a numpy.ndarray")
        if v.ndim != 2:
            raise ValueError(f"{field_name} must be 2D (grayscale), got shape {v.shape}")
        if v.size == 0:
            raise ValueError(f"{field_name} is empty")
        if v.dtype != np.uint8:
            raise ValueError(f"{field_name} dtype must be uint8, got {v.dtype}")
        return v

    @model_validator(mode='after')
    def _shapes_match(self):
        ref_shape = self.power_law.shape
        if self.hist_eq.shape != ref_shape or self.laplacian.shape != ref_shape:
            raise ValueError(
                "Shape mismatch among enhancement outputs: "
                f"power_law={ref_shape}, hist_eq={self.hist_eq.shape}, laplacian={self.laplacian.shape}"
            )
        return self
