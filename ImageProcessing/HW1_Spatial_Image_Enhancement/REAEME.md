# Homework1: Spatial Image Enhancement

## 專案概述
本專案實現了三種基本的空間域影像增強技術：直方圖均衡化、冪律（gamma）變換和拉普拉斯銳化。這些技術旨在改善影像的視覺效果、增強對比度和突顯細節。所有演算法均採用手動實現方式，不使用 OpenCV 或 PIL 中的現成增強函數，以展示對演算法的深入理解。

## 檔案結構與功能說明

### 主要檔案
- `main.py` - 程式的入口點，負責設定日誌記錄、載入測試影像，並協調整個處理流程。呼叫 `process_single_image` 函數處理每張測試影像。

### 資料夾結構

#### `/src` - 源碼資料夾
包含所有核心功能的實現程式碼，依據功能分為幾個子模組：

##### `/src/enhancement` - 增強演算法實現
- `histogram_equalization.py` - 實現了 Histogram Equalization（直方圖均衡化）演算法，通過重新分配影像的灰度值分布來增強整體對比度。包含 `HistogramEqualizationProcessor` 類和 `apply_histogram_equalization_enhancement` 函數。
- `power_law.py` - 實現了 Power Law（冪律）變換，又稱 Gamma Correction，可以調整影像的整體亮度。對於 gamma < 1 時使影像變亮，gamma > 1 時使影像變暗。包含 `PowerLawTransformer` 類和 `apply_power_law_transformation` 函數。
- `laplacian.py` - 實現了 Laplacian Sharpening（拉普拉斯銳化）濾波器，可以增強影像的細節和邊緣。提供了 4-connected 和 8-connected 兩種不同的 Laplacian kernel。包含 `LaplacianImageSharpener` 類和 `apply_laplacian_image_sharpening` 函數。

##### `/src/pipeline` - 處理管道
- `processing_pipeline.py` - 定義了完整的影像處理流程，包括運算增強效果、視覺化結果和儲存處理後的影像。主要函數有 `compute_enhancements`、`visualize_results`、`save_results` 和 `process_single_image`。

##### `/src/schemas` - 資料結構定義
- `enhancement_results_schema.py` - 定義了用於存儲所有增強結果的資料結構。

##### `/src/ui` - 使用者介面相關
- `visualization.py` - 包含 `ImageEnhancementVisualizer` 類，負責顯示原始影像和處理後的影像，以及它們對應的直方圖，以便進行視覺比較。

##### `/src/utils` - 工具函數
- `image_utils.py` - 提供了影像載入、儲存和直方圖計算等工具功能。包含 `ImageFileLoader` 和 `ImageHistogramCalculator` 兩個類。
- `logging_config.py` - 設定日誌記錄功能，輸出程式執行過程中的重要資訊。

#### `/test_image` - 測試影像
包含用於測試的標準測試影像檔案：
- `Cameraman.bmp` - 經典的攝影師灰度測試影像
- `Jetplane.bmp` - 飛機灰度測試影像
- `Lake.bmp` - 湖泊場景灰度測試影像
- `Peppers.bmp` - 彩椒灰度測試影像

#### `/results` - 結果輸出
儲存所有處理後的影像和比較圖表。執行程式後，此資料夾會包含：
- 原始和處理後的影像並排顯示
- 各種增強技術的直方圖
- 處理後的影像以 .bmp 格式儲存
- 比較圖以 .png 格式儲存

#### `/report` - 報告文件
包含專案相關的報告和文件。在此資料夾中可以找到完整的實作細節、演算法分析、實驗結果和效能比較的詳細報告，提供了對本專案實現方法和成果的深入解釋。

### 其他檔案
- `requirements.txt` - 列出專案所需的 Python 套件依賴。
- `build_exe.bat` - Windows 批次檔案，用於建立可執行檔。

## 使用方法

1. 安裝依賴：
   ```
   pip install -r requirements.txt
   ```

2. 執行程式：
   ```
   python main.py
   ```

3. 查看結果：
   執行完成後，處理過的影像和比較圖會儲存在 `results` 資料夾中，並會自動顯示。

## 演算法說明

### 1. Histogram Equalization (直方圖均衡化)
通過重新分配灰度值以增強影像的對比度。演算法步驟：
- 計算影像直方圖 (Image Histogram)
- 計算累積分布函數 (Cumulative Distribution Function, CDF)
- 根據 CDF 重新映射像素值

### 2. Power Law (Gamma) Transformation (冪律變換)
根據公式 s = c * (r/L)^gamma 調整像素值，其中：
- s 是輸出強度 (output intensity)
- r 是輸入強度 (input intensity)
- c 是縮放常數（通常為 1.0）(scaling constant)
- L 是最大強度值（uint8 為 255）(maximum intensity value)
- gamma 控制變換曲線形狀 (transformation curve parameter)

### 3. Laplacian Sharpening (拉普拉斯銳化)
使用 Laplacian filter 來增強影像的邊緣和細節：
- 應用 Laplacian kernel 進行 convolution（卷積）操作
- 將原始影像與 Laplacian response 相加
- 限制結果在 0-255 範圍內 (clamping)

## 作者資訊
- 作者：張健勳（Chien-Hsun Chang）
- 學號：614410073
- 課程：影像處理
- 學校：國立中正大學
- 日期：2025-09-29