import numpy as np


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D convolution with replicate border handling (manual loops)."""
    if image.ndim != 2:
        raise ValueError('Only 2D grayscale images are supported')
    if kernel.ndim != 2:
        raise ValueError('Kernel must be 2D')
    kernel_rows, kernel_cols = kernel.shape
    if kernel_rows % 2 == 0 or kernel_cols % 2 == 0:
        raise ValueError('Kernel dimensions must be odd for centered convolution')

    pad_r = kernel_rows // 2
    pad_c = kernel_cols // 2
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            accumulator = 0.0
            for ki in range(kernel_rows):
                for kj in range(kernel_cols):
                    ii = i + ki - pad_r
                    jj = j + kj - pad_c
                    if ii < 0:
                        ii = 0
                    elif ii >= rows:
                        ii = rows - 1
                    if jj < 0:
                        jj = 0
                    elif jj >= cols:
                        jj = cols - 1
                    accumulator += image[ii, jj] * kernel[ki, kj]
            result[i, j] = accumulator
    return result
