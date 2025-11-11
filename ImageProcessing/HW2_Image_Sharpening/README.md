## Image Processing HW #2 - Image Sharpening

This homework reproduces sharpening operations without relying on OpenCV or Pillow filters. All kernels are applied using explicit NumPy loops and FFT utilities only.

### Implemented Techniques
- Laplacian sharpening with selectable 4/8-connected kernels
- Unsharp masking using a manually defined 5x5 Gaussian smoothing kernel
- High-boost filtering (adjustable boost factor)
- Homomorphic filtering in the frequency domain (configurable parameters)

### Running the Program
```powershell
python main.py
```
The script loads grayscale images from `test_image/`, writes sharpened variants and comparison figures to `results/`, and opens a Tkinter GUI to browse the outputs.
