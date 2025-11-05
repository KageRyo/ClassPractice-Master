# HW1 Technical Documentation

## 1. Histogram Equalization

### Description
Histogram equalization is a method in image processing of contrast adjustment using the image's histogram. This technique enhances the global contrast of images by transforming the intensity distribution.

### Formula

$$
output(i,j) = \frac{255}{M \times N} \sum_{k=0}^{L-1} h(k)\cdot \sum_{l=0}^{k} h(l) 
$$

Where:
- $$h(k)$$ is the histogram of the input image.
- $$M$$ and $$N$$ are the width and height of the image.

### Usage
1. Calculate the histogram of the input image.
2. Compute the cumulative distribution function (CDF).
3. Calculate the new pixel values and construct the output image.


## 2. Contrast Stretching

### Description
Contrast stretching enhances the contrast of an image by stretching the range of intensity values it contains. It is achieved by mapping the original range of pixel values to a new range.

### Formula

$$
output(i, j) = \frac{(input(i, j) - min)}{(max - min)} \cdot (new_{max} - new_{min}) + new_{min}
$$

Where:
- $$min$$ and $$max$$ are the minimum and maximum pixel values of the input image.
- $$new_{min}$$ and $$new_{max}$$ define the new pixel value range.

### Usage
1. Identify the minimum and maximum pixel intensities in the image.
2. Map the pixel values to the new defined range.
3. Construct the enhanced image using new pixel values.


## 3. Gamma Correction

### Description
Gamma correction is a nonlinear operation used to encode and decode luminance or tristimulus values in image processing. It is beneficial for adjusting the brightness of an image without losing detail.

### Formula

$$
output(i,j) = 255 \times \left( \frac{input(i,j)}{255} \right)^{\gamma}
$$

Where:
- $$\gamma$$ is the gamma value (greater than 0).

### Usage
1. Select a suitable gamma value.
2. Apply the gamma correction formula to each pixel value in the image.
3. Reconstruct the output image with the adjusted brightness.


## Conclusion
The above techniques are essential for enhancing image quality and ensuring better visibility of relevant features. Proper application of these methods can significantly improve the outcomes in image processing tasks.