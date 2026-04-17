# Development Notes

## 0) 題目對齊

- 主指標：RMSLE
- 測試評分：排除 `visitors = 0`
- 報告輸出：使用 `Train_RMSLE` / `Test_RMSLE`
- 本專案中 cost function 與 loss function 視為同義；訓練時皆以 `MSELoss` 優化，評估時以 `RMSLE` 作為主要比較指標

## 0.1) 資料特性摘要

- visitors 分布呈長尾，且大量樣本為 `0`（店休日）
- 高流量尖峰樣本比例較低，模型容易偏向學到一般日常流量
- 因題目評分重點與營運語意，測試時排除 `visitors=0` 可避免 closed days 影響指標解讀

## 1) 調校策略總覽

1. 先修資料，再修模型。
2. 先建立穩定 baseline（MLP），再擴展序列模型（ResNet1D）。
3. 每次只改少量變數（activation、lr、sequence_length）以保留可追蹤性。
4. 以 RMSLE 為主目標，搭配 `log1p` target transform 對齊評分空間。

## 2) 主要技術改動

### 2.1 前處理與切分

- 以店鋪為單位補齊每日時間軸（reindex）
- 店休日 visitors 補 0
- Train-only StandardScaler（避免 leakage）
- 先建完整滑動序列，再依 target 年份拆 train/test

### 2.2 特徵工程

新增特徵：
- `visitors_lag_1`
- `visitors_lag_7`
- `visitors_lag_14`
- `visitors_roll_mean_7`
- `visitors_roll_std_7`

特徵維度由 9 提升到 14。

### 2.3 模型與訓練迴圈

- 最終交付主流程收斂為：MLP + ResNet1D
- 訓練優化：AdamW + ReduceLROnPlateau
- 保留 validation split 監控訓練品質
- 移除 early stopping 機制，固定跑滿指定 epochs
- 支援 target transform：`none` / `log1p`
- 支援 peak-weighted loss：`peak_weight` + `peak_quantile`
- GPU/CPU 自動切換（訓練與推論）

## 3) 關鍵觀察

1. ReLU 輸出層在此任務曾導致偏零預測，改 Softplus 後穩定性更好。
2. lag/rolling 對 RMSLE 與泛化有明顯幫助。
3. ResNet1D 對 learning rate 較敏感，`2e-4` 在目前最終配置下較穩定。
4. 某些設定會出現 RMSLE 進步但 R2 一般，代表尾部樣本誤差仍需改善。

## 4) 目前實驗結論（截至 2026-04-16）

1. `mlp` 是穩定且快速的 baseline。
2. `resnet1d` 在當前特徵與訓練策略下具更高上限。
3. 最終交付版固定使用 `mlp,resnet1d`，降低流程複雜度並提升可重現性。

## 4.1) 最終提交結果（鎖定）

- Best Model: `resnet1d`
- Train RMSLE: `0.558909`
- Test RMSLE: `0.560029`
- Train RMSE: `9.927783`
- Test RMSE: `23.959431`
- Train R2: `0.633555`
- Test R2: `-0.973978`
- Overfit Gap: `0.001120`（Overfitting Risk: `No`）

此結果對應目前 `results.csv` 與 `best_model_summary.md`，本專案不再進行新訓練，後續僅維持文件與程式可重現性。

## 5) 推薦執行流程

### 5.1 主要比較

```bash
python main.py
```

`python main.py` 預設即對齊最終建議設定（`mlp,resnet1d`、100 epochs、`sequence_length=14`、`target_transform=log1p`、`resnet_lr=0.0002`、預設不繪圖）。

### 5.2 正式交付執行（固定跑滿 epoch）

```bash
python main.py --models mlp,resnet1d --mlp-epochs 100 --resnet-epochs 100 --sequence-length 14 --mlp-lr 0.001 --resnet-lr 0.0002 --target-transform log1p --val-start-date 2016-10-01 --nn-log-interval 5
```

## 6) 里程碑（Timeline）

1. Phase 1: 修正資料管線（補日、補 0、train-only scaling）
2. Phase 2: 擴展模型族（LSTM/CNN/Transformer/ResNet1D）
3. Phase 3: 訓練強化（AdamW、scheduler、GPU）
4. Phase 4: 穩定性修正（Softplus 輸出、調整 ResNet lr）
5. Phase 5: 特徵工程升級（lag/rolling）
6. Phase 6: 泛化策略（validation + log1p）
7. Phase 7: 最終交付收斂（移除 early stopping 與搜尋腳本）

## 7) 下一輪優化（依優先順序）

1. 固定最終配置後以 3 個 seed 重跑，回報 mean/std
2. 若仍需提升，優先新增特徵（reservation lag、holiday lead/lag）
3. 最後才考慮增加模型深度或更長訓練 epoch
