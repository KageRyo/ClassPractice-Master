# Homework 1：Spatial Image Enhancement

## 專案簡介
此作業實作三種經典的空間域影像增強技術：直方圖均衡化、冪律（Gamma）變換與拉普拉斯銳化。所有演算法皆以純 NumPy 實作，不呼叫 OpenCV / PIL 現成增強函式，藉此強化對演算法細節的掌握。系統包含批次處理流程、結果視覺化以及 Tkinter 互動式檢視介面。

## 系統架構總覽
- `main.py`：程式進入點，負責初始化日誌、載入測試影像、啟動背景處理執行緒與 Tkinter 檢視介面。
- `src/pipeline/processing_pipeline.py`：定義核心處理步驟，負責執行三種增強演算法、產生比較圖與直方圖並儲存結果。
- `src/enhancement/`：三個增強模組（直方圖均衡化、冪律變換、拉普拉斯銳化）的實作。
- `src/utils/`：影像載入與儲存、直方圖計算，以及日誌設定等共用工具。
- `src/ui/visualization.py`：以 Matplotlib 繪製結果圖與直方圖。
- `src/ui/gui.py`：Tkinter 介面，可在處理完成後逐張瀏覽比較圖、檢視自動計算的 Gamma 值並查看執行紀錄。
- `src/schemas/enhancement_results_schema.py`：以 Pydantic 驗證增強結果，確保三種輸出影像的形狀與資料型態一致。

下圖為主要資料流：

1. `ImageFileLoader` 讀取 `test_image/` 中的影像。
2. `process_single_image` 先決定 Gamma，接著透過三種演算法生成增強結果。
3. `ImageEnhancementVisualizer` 產生比較圖及各自的直方圖。
4. 所有輸出（影像與圖表）存放於 `results/`，並由 `ImageReviewApp` 提供互動式檢視。

## 目錄結構
```
HW1_Spatial_Image_Enhancement/
├─ main.py
├─ pyproject.toml
├─ requirements.txt
├─ build_exe.bat
├─ HW1_Image_Enhancement.spec
├─ src/
│  ├─ enhancement/
│  │  ├─ histogram_equalization.py
│  │  ├─ laplacian.py
│  │  └─ power_law.py
│  ├─ pipeline/processing_pipeline.py
│  ├─ schemas/enhancement_results_schema.py
│  ├─ ui/
│  │  ├─ gui.py
│  │  └─ visualization.py
│  └─ utils/
│     ├─ image_utils.py
│     └─ logging_config.py
├─ test_image/
└─ results/
```

## 增強演算法摘要
- **Histogram Equalization**：計算影像直方圖與累積分布函數 (CDF)，重新映射像素值以提升整體對比。
- **Power-Law (Gamma) Transformation**：採用公式 `s = c * (r/L)^gamma`，透過 Gamma 調整影像亮度。若未指定 Gamma，系統會估算合適的值。
- **Laplacian Sharpening**：使用 4 或 8 鄰域的 Laplacian kernel 進行卷積後與原圖相加，藉此強化邊緣細節。

## 建置與執行
### 安裝流程（使用 `pyproject.toml`）
```bash
python -m pip install --upgrade pip
python -m pip install .
```
### 執行批次處理與檢視介面
```bash
python main.py
```

成功執行後，`results/` 會產生：
- 三種增強結果（BMP）
- 各增強方法的直方圖（PNG）
- 原圖與增強結果的並排比較圖（PNG）

## Tkinter 檢視介面說明
- 介面主畫面顯示選定影像的比較圖，下方提供處理過程的日誌。
- 使用 `Previous` / `Next` 或下拉選單切換影像，狀態列會顯示當前索引與使用的 Gamma 值（自動或手動）。
- 視窗大小變更時會自動重新調整比較圖，確保完整顯示。

## 打包為可執行檔
建議在乾淨的虛擬環境安裝專案後執行：
```bash
python -m pip install ".[build]"
& .\build_exe.bat
```
或手動執行：
```bash
python -m PyInstaller --clean --noconfirm HW1_Image_Enhancement.spec
```
打包完成後，`dist/HW1_Image_Enhancement.exe` 可直接在 Windows 上執行，輸出仍會寫入當前目錄的 `results/`。

## 作者資訊
- 張健勳（Chien-Hsun Chang）
- 國立中正大學影像處理課程作業 #1
- 學號：614410073
- 日期：2025-11-01