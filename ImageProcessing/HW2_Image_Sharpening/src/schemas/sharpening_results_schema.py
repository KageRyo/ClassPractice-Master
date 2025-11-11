import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SharpeningResultsSchema(BaseModel):
    """Validated container for sharpening outputs (uint8, 2D, matching shapes)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    laplacian: np.ndarray = Field(..., description='Laplacian sharpened image array')
    unsharp_mask: np.ndarray = Field(..., description='Unsharp masking result array')
    high_boost: np.ndarray = Field(..., description='High-boost filtered image array')
    homomorphic: np.ndarray = Field(..., description='Homomorphic filtered image array')

    @field_validator('laplacian', 'unsharp_mask', 'high_boost', 'homomorphic', mode='before')
    @classmethod
    def validate_array_contents(cls, value, info):
        field_name = info.field_name
        if not isinstance(value, np.ndarray):
            raise TypeError(f'{field_name} must be a numpy.ndarray')
        if value.ndim != 2:
            raise ValueError(f'{field_name} must be 2D (grayscale), got shape {value.shape}')
        if value.size == 0:
            raise ValueError(f'{field_name} is empty')
        if value.dtype != np.uint8:
            raise ValueError(f'{field_name} dtype must be uint8, got {value.dtype}')
        return value

    @model_validator(mode='after')
    def ensure_matching_shapes(self):
        reference_shape = self.laplacian.shape
        if (
            self.unsharp_mask.shape != reference_shape
            or self.high_boost.shape != reference_shape
            or self.homomorphic.shape != reference_shape
        ):
            raise ValueError(
                'Shape mismatch among sharpening outputs: '
                f"laplacian={reference_shape}, "
                f"unsharp={self.unsharp_mask.shape}, "
                f"high_boost={self.high_boost.shape}, "
                f"homomorphic={self.homomorphic.shape}"
            )
        return self
