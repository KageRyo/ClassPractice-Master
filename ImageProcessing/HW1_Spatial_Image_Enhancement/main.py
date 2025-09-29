import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load test images
image_names = ['Cameraman.bmp', 'Jetplane.bmp', 'Lake.bmp', 'Peppers.bmp']
images = {}

print("Loading test images...")
for name in image_names:
    img_path = f'test_image/{name}'
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    images[name] = np.array(img, dtype=np.float64)
    print(f"Loaded {name}: {images[name].shape}")

# Process each image with three enhancement techniques
for img_name in image_names:
    original_img = images[img_name]
    print(f"\nProcessing {img_name}...")
    
    # Create figure for displaying results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Image Enhancement Results - {img_name}', fontsize=16)
    
    # Display original image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # (1) Power-law (gamma) transformation
    print("  Applying power-law transformation...")
    gamma = 2.2  # Gamma value for enhancement
    c = 1.0      # Scaling constant
    rows, cols = original_img.shape
    
    # Apply power-law transformation manually: s = c * r^gamma
    gamma_result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            # Normalize pixel to [0,1] range
            normalized_pixel = original_img[i, j] / 255.0
            # Apply power-law transformation
            gamma_pixel = c * (normalized_pixel ** gamma)
            # Scale back to [0,255] and ensure valid range
            result_pixel = gamma_pixel * 255.0
            if result_pixel > 255:
                result_pixel = 255
            elif result_pixel < 0:
                result_pixel = 0
            gamma_result[i, j] = int(result_pixel)
    
    # Display gamma transformation result
    axes[0, 1].imshow(gamma_result, cmap='gray')
    axes[0, 1].set_title(f'Power-law (γ={gamma})')
    axes[0, 1].axis('off')
    
    # Save gamma transformation result
    gamma_img = Image.fromarray(gamma_result)
    gamma_img.save(f'results/{img_name[:-4]}_gamma.bmp')
    
    # (2) Histogram equalization
    print("  Applying histogram equalization...")
    
    # Calculate histogram manually
    histogram = [0] * 256  # Initialize histogram with zeros
    total_pixels = rows * cols
    
    # Count frequency of each pixel intensity
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(original_img[i, j])
            histogram[pixel_value] += 1
    
    # Calculate cumulative distribution function (CDF) manually
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for k in range(1, 256):
        cdf[k] = cdf[k-1] + histogram[k]
    
    # Find minimum non-zero CDF value
    cdf_min = 0
    for k in range(256):
        if cdf[k] > 0:
            cdf_min = cdf[k]
            break
    
    # Apply histogram equalization transformation manually
    hist_eq_result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(original_img[i, j])
            # Apply histogram equalization formula
            new_value = ((cdf[pixel_value] - cdf_min) * 255) / (total_pixels - cdf_min)
            if new_value > 255:
                new_value = 255
            elif new_value < 0:
                new_value = 0
            hist_eq_result[i, j] = int(new_value)
    
    # Display histogram equalization result
    axes[0, 2].imshow(hist_eq_result, cmap='gray')
    axes[0, 2].set_title('Histogram Equalization')
    axes[0, 2].axis('off')
    
    # Save histogram equalization result
    hist_eq_img = Image.fromarray(hist_eq_result)
    hist_eq_img.save(f'results/{img_name[:-4]}_hist_eq.bmp')
    
    # (3) Image sharpening using Laplacian operator
    print("  Applying Laplacian sharpening...")
    
    # Define Laplacian kernel (8-connected)
    laplacian_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])
    
    # Apply Laplacian filter using convolution
    rows, cols = original_img.shape
    laplacian_result = np.zeros_like(original_img)
    
    # Pad the image for convolution
    padded_img = np.pad(original_img, ((1, 1), (1, 1)), mode='edge')
    
    # Perform convolution manually without using np.sum
    for i in range(rows):
        for j in range(cols):
            # Handle boundary conditions by using edge pixels
            laplacian_response = 0.0
            for ki in range(-1, 2):  # -1, 0, 1
                for kj in range(-1, 2):  # -1, 0, 1
                    # Calculate image coordinates
                    img_i = i + ki
                    img_j = j + kj
                    
                    # Handle boundaries by clamping
                    if img_i < 0:
                        img_i = 0
                    elif img_i >= rows:
                        img_i = rows - 1
                    if img_j < 0:
                        img_j = 0
                    elif img_j >= cols:
                        img_j = cols - 1
                    
                    # Apply kernel
                    kernel_value = laplacian_kernel[ki + 1, kj + 1]
                    laplacian_response += original_img[img_i, img_j] * kernel_value
            
            laplacian_result[i, j] = laplacian_response
    
    # Add Laplacian result to original image for sharpening manually
    sharpened_img = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            new_value = original_img[i, j] + laplacian_result[i, j]
            # Clip values manually to valid range [0, 255]
            if new_value > 255:
                new_value = 255
            elif new_value < 0:
                new_value = 0
            sharpened_img[i, j] = int(new_value)
    
    # Display Laplacian sharpening result
    axes[0, 3].imshow(sharpened_img, cmap='gray')
    axes[0, 3].set_title('Laplacian Sharpening')
    axes[0, 3].axis('off')
    
    # Save Laplacian sharpening result
    sharp_img = Image.fromarray(sharpened_img)
    sharp_img.save(f'results/{img_name[:-4]}_sharpened.bmp')
    
    # Display histograms manually calculated
    
    # Calculate histogram for original image (already calculated above)
    x_values = list(range(256))
    axes[1, 0].bar(x_values, histogram, alpha=0.7, color='blue')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    
    # Calculate histogram for gamma result
    gamma_hist = [0] * 256
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(gamma_result[i, j])
            gamma_hist[pixel_value] += 1
    axes[1, 1].bar(x_values, gamma_hist, alpha=0.7, color='green')
    axes[1, 1].set_title('Power-law Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    
    # Calculate histogram for histogram equalization result
    hist_eq_hist = [0] * 256
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(hist_eq_result[i, j])
            hist_eq_hist[pixel_value] += 1
    axes[1, 2].bar(x_values, hist_eq_hist, alpha=0.7, color='red')
    axes[1, 2].set_title('Equalized Histogram')
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    
    # Calculate histogram for sharpened image
    sharp_hist = [0] * 256
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(sharpened_img[i, j])
            sharp_hist[pixel_value] += 1
    axes[1, 3].bar(x_values, sharp_hist, alpha=0.7, color='orange')
    axes[1, 3].set_title('Sharpened Histogram')
    axes[1, 3].set_xlabel('Pixel Intensity')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'results/{img_name[:-4]}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Results saved for {img_name}")

print("\nAll image processing completed!")
print("Results saved in 'results/' directory")
print("- Original and processed images displayed")
print("- Histograms shown for all enhancement techniques")
print("- All processed images saved as .bmp files")