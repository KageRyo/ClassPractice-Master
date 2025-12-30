"""
Color Enhancement Results Schema

Pydantic schema for validating color image enhancement results.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ColorEnhancementResultsSchema(BaseModel):
    """Validated color enhancement result container (Pydantic v2).

    Validation:
    - Each field: 3D, non-empty, uint8 ndarray with shape (H, W, 3)
    - All shapes identical
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgb_histogram_eq: np.ndarray = Field(..., description="RGB histogram equalized image")
    hsi_histogram_eq: np.ndarray = Field(..., description="HSI intensity histogram equalized image")
    hsi_intensity_contrast: np.ndarray = Field(..., description="HSI intensity contrast stretched image")
    hsi_gamma: np.ndarray = Field(..., description="HSI intensity gamma corrected image")
    hsi_saturation: np.ndarray = Field(..., description="HSI saturation enhanced image")

    @field_validator(
        'rgb_histogram_eq',
        'hsi_histogram_eq',
        'hsi_intensity_contrast',
        'hsi_gamma',
        'hsi_saturation',
        mode='before'
    )
    @classmethod
    def validate_array_contents(cls, v, info):
        field_name = info.field_name
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{field_name} must be a numpy.ndarray")
        if v.ndim != 3:
            raise ValueError(f"{field_name} must be 3D (color image), got shape {v.shape}")
        if v.shape[2] != 3:
            raise ValueError(f"{field_name} must have 3 channels (RGB), got {v.shape[2]}")
        if v.size == 0:
            raise ValueError(f"{field_name} is empty")
        if v.dtype != np.uint8:
            raise ValueError(f"{field_name} dtype must be uint8, got {v.dtype}")
        return v

    @model_validator(mode='after')
    def ensure_matching_shapes(self):
        ref_shape = self.rgb_histogram_eq.shape
        for field_name in ['hsi_histogram_eq', 'hsi_intensity_contrast', 'hsi_gamma', 'hsi_saturation']:
            field_value = getattr(self, field_name)
            if field_value.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {field_name} has shape {field_value.shape}, "
                    f"expected {ref_shape}"
                )
        return self
