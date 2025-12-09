# HW3 - Image Restoration

Image Processing Homework #3 - 影像復原

## 題目說明

給定四張灰階影像及其對應的退化影像（加入零均值、標準差為10的高斯隨機雜訊），使用以下兩種方法估計原始影像 f̂(x,y)：

1. **Direct Inverse Filtering with Low-pass Filter** - 直接逆濾波（結合低通濾波器）
2. **Minimum Mean-Square Error (Wiener) Filtering** - 最小均方誤差（維納）濾波

### 退化函數

退化函數定義為：

$$H(u,v) = e^{-k(u^2+v^2)^{5/6}}$$

其中 k 為系統參數。