# HW4 - Color Image Enhancement

## 作業說明

本作業實現彩色影像增強技術，在 RGB 和 HSI 色彩空間中進行影像增強處理。

### 需求
- 給定六張彩色影像（4張 640x500 和 2張 357x535 的 PNG 檔）
- 在 RGB 和 HSI 色彩空間中增強彩色影像
- Hue (H) 成分不能改變
- 所有最終結果需轉換回 RGB 色彩空間顯示
- **不使用 OpenCV 或 PIL 的影像處理函式**

## 專案結構

```
HW4_Color_Image_Enhancement/
├── main.py                     # 主程式入口
├── pyproject.toml              # 專案設定
├── README.md                   # 說明文件
├── build_exe.bat              # PyInstaller 打包腳本
├── test_image/                 # 測試圖片資料夾
├── results/                    # 輸出結果資料夾
├── pyinstaller_hooks/          # PyInstaller hooks
└── src/
    ├── __init__.py
    ├── color_space/           # 色彩空間轉換
    │   ├── __init__.py
    │   └── color_conversion.py  # RGB <-> HSI 轉換
    ├── enhancement/           # 影像增強演算法
    │   ├── __init__.py
    │   ├── histogram_equalization.py  # 直方圖等化
    │   ├── gamma_correction.py        # Gamma 校正
    │   ├── saturation_enhancement.py  # 飽和度增強
    │   └── intensity_enhancement.py   # 強度對比拉伸
    ├── pipeline/              # 處理流程
    │   ├── __init__.py
    │   └── processing_pipeline.py
    ├── schemas/               # 資料結構驗證
    │   ├── __init__.py
    │   └── enhancement_results_schema.py
    ├── ui/                    # 使用者介面
    │   ├── __init__.py
    │   ├── gui.py             # Tkinter GUI
    │   └── visualization.py   # Matplotlib 視覺化
    └── utils/                 # 工具函式
        ├── __init__.py
        ├── image_utils.py     # 影像載入/儲存
        └── logging_config.py  # 日誌設定
```

## 實現的增強技術

### 1. RGB 色彩空間增強
- **RGB Histogram Equalization**: 對 R、G、B 三個通道分別進行直方圖等化

### 2. HSI 色彩空間增強（保持 Hue 不變）
- **HSI Intensity Histogram Equalization**: 僅對 Intensity 通道進行直方圖等化
- **HSI Intensity Gamma Correction**: 對 Intensity 通道進行 Gamma 校正
- **HSI Saturation Enhancement**: 增強 Saturation 飽和度

## 使用方式

### 執行程式（含 GUI）
```bash
python main.py
```

### 執行程式（無 GUI，批次處理）
```bash
python main.py --no-gui
```

### 建置執行檔
```bash
build_exe.bat
```

## 技術說明

### RGB to HSI 轉換
使用 Gonzalez & Woods 教科書中的公式：
- $I = \frac{R + G + B}{3}$
- $S = 1 - \frac{3 \cdot \min(R, G, B)}{R + G + B}$
- $H = \cos^{-1}\left(\frac{\frac{1}{2}[(R-G)+(R-B)]}{\sqrt{(R-G)^2 + (R-B)(G-B)}}\right)$

### HSI to RGB 轉換
根據 Hue 所在的扇區（RG、GB、BR）使用對應的轉換公式。

### 直方圖等化
手動計算直方圖、累積分佈函數（CDF），並應用等化映射。

### Gamma 校正
$I_{out} = I_{in}^{\gamma}$

其中 $\gamma < 1$ 會使影像變亮，$\gamma > 1$ 會使影像變暗。

## 依賴套件
- numpy
- pillow（僅用於影像讀取/儲存）
- matplotlib
- pydantic>=2.0

## 作者
張健勳 (614410073)
中正大學 影像處理課程
