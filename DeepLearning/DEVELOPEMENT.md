# Development Notes

## 0) 題目對齊

- 主指標：RMSLE
- 測試評分：排除 `visitors = 0`
- 報告輸出：使用 `Train_RMSLE` / `Test_RMSLE`

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

- 新增序列模型：LSTM、CNN1D、Transformer、ResNet1D
- 訓練優化：AdamW + ReduceLROnPlateau
- 支援 validation split + early stopping（可關閉）
- 支援 `min_epochs_before_stop`
- 支援 target transform：`none` / `log1p`
- 支援 peak-weighted loss：`peak_weight` + `peak_quantile`
- GPU/CPU 自動切換（訓練與推論）

## 3) 關鍵觀察

1. ReLU 輸出層在此任務曾導致偏零預測，改 Softplus 後穩定性更好。
2. lag/rolling 對 RMSLE 與泛化有明顯幫助。
3. ResNet1D 對 learning rate 較敏感，`3e-4` 比 `1e-3` 更穩。
4. 某些設定會出現 RMSLE 進步但 R2 一般，代表尾部樣本誤差仍需改善。

## 4) 目前實驗結論（截至 2026-04-16）

1. `mlp` 是穩定且快速的 baseline。
2. `resnet1d` 在當前特徵與訓練策略下具更高上限。
3. 實務上優先比較 `mlp,resnet1d`，再擴展到其他模型。

## 5) 推薦執行流程

### 5.1 主要比較

```bash
python main.py --models mlp,resnet1d --skip-plot --target-transform log1p
```

### 5.2 跑滿 epoch（關閉 early stopping）

```bash
python main.py --models mlp,resnet1d --mlp-epochs 100 --resnet-epochs 100 --disable-early-stopping --skip-plot
```

### 5.3 小型網格搜尋

```bash
bash scripts/grid_search_mlp_resnet.sh
```

## 6) 里程碑（Timeline）

1. Phase 1: 修正資料管線（補日、補 0、train-only scaling）
2. Phase 2: 擴展模型族（LSTM/CNN/Transformer/ResNet1D）
3. Phase 3: 訓練強化（AdamW、scheduler、GPU）
4. Phase 4: 穩定性修正（Softplus 輸出、調整 ResNet lr）
5. Phase 5: 特徵工程升級（lag/rolling）
6. Phase 6: 泛化策略（validation + early stopping + log1p）
7. Phase 7: 自動化搜尋（grid script）

## 7) 下一輪優化（依優先順序）

1. 搜尋 `sequence_length`（7/14/21/28）
2. 分開搜尋 MLP 與 ResNet1D learning rate
3. 調整 peak-weighted loss，提升高峰樣本表現
4. 多 seed 重跑（>= 3）並回報 mean/std
