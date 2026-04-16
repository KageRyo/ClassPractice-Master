# Deep Learning Midterm - Restaurant Visitors Forecasting

本專案為期中考程式實作，主評估指標為 RMSLE（Root Mean Squared Log Error）。

## 1) 環境

- Python: 3.11
- Conda env: `dl-class`

```bash
conda activate dl-class
pip install -r requirements.txt
```

## 2) 專案結構

- `main.py`: 主訓練與評估入口
- `src/preprocessing/`: 前處理、序列切分、scaler
- `src/models/`: 模型定義與訓練
- `src/evaluation/`: 指標評估與結果輸出
- `results.csv`: 每次訓練結果彙整
- `best_model_summary.md`: 最佳模型摘要

## 3) 快速執行

預設只跑 `mlp,resnet1d`：

```bash
python main.py --skip-plot
```

完整 100 epochs（關閉 early stopping）：

```bash
python main.py \
  --models mlp,resnet1d \
  --mlp-epochs 100 \
  --resnet-epochs 100 \
  --sequence-length 14 \
  --mlp-lr 0.001 \
  --resnet-lr 0.0002 \
  --target-transform log1p \
  --val-start-date 2016-10-01 \
  --disable-early-stopping \
  --skip-plot \
  --nn-log-interval 5
```

## 4) 重要參數

- `--models`: 模型清單（逗號分隔）
- `--sequence-length`: 序列長度
- `--target-transform`: `none` 或 `log1p`
- `--mlp-lr`, `--resnet-lr`: 學習率
- `--early-stopping-patience`, `--min-epochs-before-stop`
- `--disable-early-stopping`
- `--peak-weight`, `--peak-quantile`: 高峰樣本加權

## 5) 目前重點設定

- 預設比較模型：`mlp` vs `resnet1d`
- 測試評分排除 `visitors = 0`（closed days）
- 神經模型支援 validation split、early stopping、log1p target transform、GPU 自動切換

## 6) 自動化搜尋

```bash
bash scripts/grid_search_mlp_resnet.sh
```

輸出於 `grid_search_runs/grid_summary.csv`。

## 7) 文件分工

- README: 快速上手與執行指令
- `DEVELOPEMENT.md`: 完整調校歷程、方法原因、觀察與下一步
