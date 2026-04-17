# Final Submission Guide

## 1) 最終執行指令

```bash
conda activate dl-class
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

## 2) 交付前檢查

- 確認 `results.csv` 已更新且有 `mlp`、`resnet1d` 兩行
- 確認 `best_model_summary.md` 已更新
- 確認 `models/` 內有本次訓練輸出的 `.pth` 檔
- 確認 README 與 DEVELOPEMENT 文件內容與程式一致

## 3) 報告可直接引用欄位

- 主指標：`Test_RMSLE`
- 輔助指標：`Test_RMSE`, `Test_MAE`, `Test_R2`, `Test_Peak_Recall`
- 風險指標：`Overfit_Gap`, `Overfitting_Flag`

## 4) 最終版範圍

- 主入口：`main.py`
- 核心模型：`mlp`, `resnet1d`
- 已移除：early stopping 機制、搜尋腳本流程
