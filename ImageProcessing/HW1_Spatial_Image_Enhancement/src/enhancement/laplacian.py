import numpy as np
import logging

logger = logging.getLogger(__name__)

class LaplacianImageSharpener:
    """Apply Laplacian filtering and create a sharpened image."""

    def __init__(self, laplacian_kernel_type='8-connected'):
        """Select Laplacian kernel type: '8-connected' (default) or '4-connected'."""
        self.laplacian_kernel_type = laplacian_kernel_type
        self.laplacian_kernel = self._get_laplacian_kernel(laplacian_kernel_type)
        logger.info(f"LaplacianImageSharpener initialized with {laplacian_kernel_type} kernel")

    def _get_laplacian_kernel(self, laplacian_kernel_type):
        """Return Laplacian kernel matrix for requested connectivity."""
        if laplacian_kernel_type == '8-connected':
            return np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float64)
        elif laplacian_kernel_type == '4-connected':
            return np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=np.float64)
        else:
            raise ValueError(f"Unknown kernel type: {laplacian_kernel_type}")
    
    def apply_laplacian_convolution_filter(self, image):
        """Return raw Laplacian response via explicit zero-padding at borders."""
        rows, cols = image.shape
        laplacian_result = np.zeros_like(image, dtype=np.float64)
        logger.debug(f"Applying Laplacian convolution to {rows}x{cols} image")
        for i in range(rows):
            for j in range(cols):
                laplacian_response = 0.0
                for ki in range(-1, 2):  # -1, 0, 1
                    for kj in range(-1, 2):  # -1, 0, 1
                        img_i = i + ki
                        img_j = j + kj
                        if img_i < 0:
                            img_i = 0
                        elif img_i >= rows:
                            img_i = rows - 1
                        if img_j < 0:
                            img_j = 0
                        elif img_j >= cols:
                            img_j = cols - 1
                        laplacian_kernel_value = self.laplacian_kernel[ki + 1, kj + 1]
                        laplacian_response += image[img_i, img_j] * laplacian_kernel_value
                laplacian_result[i, j] = laplacian_response
        logger.debug("Laplacian convolution completed")
        return laplacian_result

    def apply_sharpening_filter(self, image):
        """Return image + Laplacian response (clamped to 0..255)."""
        rows, cols = image.shape
        logger.debug(f"Sharpening {rows}x{cols} image with Laplacian")
        laplacian_filtered_result = self.apply_laplacian_convolution_filter(image)
        sharpened_image_result = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                enhanced_pixel_value = image[i, j] + laplacian_filtered_result[i, j]
                if enhanced_pixel_value > 255:
                    enhanced_pixel_value = 255
                elif enhanced_pixel_value < 0:
                    enhanced_pixel_value = 0
                sharpened_image_result[i, j] = int(enhanced_pixel_value)
        logger.debug("Laplacian sharpening completed")
        return sharpened_image_result

def apply_laplacian_image_sharpening(image, laplacian_kernel_type='8-connected'):
    """Functional wrapper returning sharpened image for one-off calls."""
    image_sharpener = LaplacianImageSharpener(laplacian_kernel_type)
    return image_sharpener.apply_sharpening_filter(image)