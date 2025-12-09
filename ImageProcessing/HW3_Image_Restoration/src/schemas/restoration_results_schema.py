import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RestorationResultsSchema(BaseModel):
    """Validated container for restoration outputs (uint8, 2D, matching shapes)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inverse_filtered: np.ndarray = Field(..., description='Inverse filtered image array')
    wiener_filtered: np.ndarray = Field(..., description='Wiener filtered image array')

    @field_validator('inverse_filtered', 'wiener_filtered', mode='before')
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
        reference_shape = self.inverse_filtered.shape
        if self.wiener_filtered.shape != reference_shape:
            raise ValueError(
                'Shape mismatch among restoration outputs: '
                f"inverse_filtered={reference_shape}, "
                f"wiener_filtered={self.wiener_filtered.shape}"
            )
        return self
