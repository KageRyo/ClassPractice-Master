# Deep Learning Midterm - Restaurant Visitors Forecasting

本專案最終交付版本以 RMSLE 為主指標，並固定主流程為 `mlp` 與 `resnet1d` 兩個模型比較。

## 1) 環境

- Python: 3.11
- Conda env: `dl-class`

```bash
conda activate dl-class
pip install -r requirements.txt
```

## 2) 一次跑完整最終版本

最簡單：

```bash
python main.py
```

`python main.py` 現在預設即對齊最終建議設定（`mlp,resnet1d`、100 epochs、`sequence_length=14`、`mlp_lr=0.001`、`resnet_lr=0.0002`、`target_transform=log1p`、`val_start_date=2016-10-01`、預設不繪圖）。

等價完整指令：

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
  --nn-log-interval 5
```

## 3) 輸出檔案

- `results.csv`: 本次所有模型評估結果
- `best_model_summary.md`: 最佳模型摘要，可直接填報告
- `models/`: 儲存模型權重檔

## 3.1) 最終鎖定結果（不再重跑）

- Best Model: `resnet1d`
- Train RMSLE: `0.558909`
- Test RMSLE: `0.560029`
- Train R2: `0.633555`
- Test R2: `-0.973978`
- Test Peak Recall: `0.446825`

以上數值來自最終版 `results.csv` 與 `best_model_summary.md`，作為本次交付固定結果。

## 4) 目前保留的核心參數

- `--models`: 僅支援 `mlp,resnet1d`
- `--mlp-hidden-size`
- `--mlp-epochs`, `--resnet-epochs`
- `--sequence-length`
- `--mlp-lr`, `--resnet-lr`
- `--target-transform` (`none` or `log1p`)
- `--val-start-date`
- `--peak-weight`, `--peak-quantile`

## 5) 文件對照

- `README.md`: 交付版快速執行說明
- `DEVELOPEMENT.md`: 方法、變更歷程、實驗觀察
- `FINAL_SUBMISSION_GUIDE.md`: 最終繳交前檢查清單
