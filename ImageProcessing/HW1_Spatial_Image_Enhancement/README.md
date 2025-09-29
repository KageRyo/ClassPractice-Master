# HW1：Spatial Image Enhancement

本資料夾提供影像處理課程第一次作業所需的檔案結構，並說明如何在 Windows 環境下執行 `main.py` 以完成下列三種空域影像強化操作：

1. Power-law（Gamma）變換
2. Histogram Equalization
3. Laplacian 影像銳化

## 執行環境與相依套件

- **作業系統**：Windows 10 以上（助教端不需額外安裝第三方軟體即可執行）
- **Python**：建議使用 Python 3.9～3.12
- **必要套件**（已列於 `requirements.txt`）
	- `numpy`
	- `pillow`
	- `matplotlib`
	- `pyinstaller`（僅於打包成可執行檔時需要）

### 安裝步驟

```powershell
# 建議先建立虛擬環境
python -m venv .venv
.\.venv\Scripts\activate

# 安裝必要套件
pip install --upgrade pip
pip install -r requirements.txt
```

## 專案結構

```
HW1_Spatial_Image_Enhancement/
├── main.py               # 作業主程式（需自行完成演算法邏輯）
├── build_exe.bat         # 打包成 Windows 可執行檔的批次檔
├── requirements.txt      # Python 套件清單
├── images/               # 測試影像（256×256 BMP）
├── results/              # 建議放置輸出影像與圖表
└── README.md             # 使用說明（本文件）
```

## 建立 Windows 可執行檔
使用隨附的批次檔打包：

```powershell
build_exe.bat
```

執行後會在 `dist/` 產生 `HW1_Image_Enhancement.exe`。

## 常見問題

| 問題 | 排除方式 |
| --- | --- |
| 執行時看不到視窗或圖像 | 確認 `matplotlib` 版本正確，並檢查程式是否使用 `plt.show()` 或將結果儲存至 `results/`。 |
| 找不到輸入影像 | 確保 `images/` 目錄存在且檔名正確無中文或空白。 |
| 可執行檔無法輸出結果 | 確認打包後的資料夾仍保有 `images/`、`results/`，並於程式內使用相對路徑讀寫。 |