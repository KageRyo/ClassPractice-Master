1. 題目公告說明在 README.md 中
2. 資料集下載自 eCourse2 後放到 datasets 資料夾
3. 題目需求在 Mid-Term Programming Exam.md
---
```
conda activate dl-class
```
- Python 3.11

預設執行：

```bash
python main.py --skip-plot
```

- 預設只會跑 `mlp,resnet1d`。
- 若要覆寫模型清單，請使用 `--models`，例如：

```bash
python main.py --models lgbm,mlp,lstm,cnn1d,resnet1d,transformer --skip-plot
```

近期調校流程：

1. 先固定資料切分，再進行特徵工程（lag/rolling）與輸出層穩定化（Softplus）。
2. 神經網路訓練預設採用 validation + early stopping。
3. target transform 預設使用 `log1p`，可透過參數切換。

建議實驗指令：

```bash
python main.py --models mlp,resnet1d --mlp-epochs 100 --resnet-epochs 100 --skip-plot --target-transform log1p --val-start-date 2016-10-01 --early-stopping-patience 12
```

完整跑滿 100 epochs（關閉 early stopping）：

```bash
python main.py --models mlp,resnet1d --mlp-epochs 100 --resnet-epochs 100 --skip-plot --target-transform log1p --val-start-date 2016-10-01 --disable-early-stopping
```

若要關閉 target transform：

```bash
python main.py --models mlp,resnet1d --skip-plot --target-transform none
```

小型超參數搜尋（sequence_length x lr）：

```bash
bash scripts/grid_search_mlp_resnet.sh
```

---

調校過程紀錄（可用於報告）：

1. 初期問題：
	- 序列資料有缺日（店休未記錄），模型難以學習穩定時序規律。
	- 缺乏一致縮放，神經網路在不同尺度特徵下收斂速度慢。
	- 深層序列模型早期出現 underfitting 或不穩定。

2. 資料層改善：
	- 每店鋪日期補齊（reindex），缺失 visitors 補 0。
	- 連續特徵使用 train-only StandardScaler，避免資料洩漏。
	- 新增 temporal features：`visitors_lag_1/7/14`、`visitors_roll_mean_7/std_7`。

3. 模型層改善：
	- MLP 與 ResNet1D 輸出改為 Softplus（保留非負且避免 ReLU 硬截斷）。
	- ResNet1D 採較小 learning rate（預設 `3e-4`）提升穩定性。
	- 訓練迴圈加入 GPU 自動切換、validation 監控與可控 early stopping。

4. 策略層改善：
	- target transform 預設 `log1p`，與 RMSLE 目標更一致。
	- early stopping 可關閉，或設定最小啟動 epoch，避免過早停止。

為何最後聚焦比較 MLP vs ResNet1D：

1. MLP：
	- 訓練快、穩定、可作為強基準線。
	- 對時間衍生特徵（lag/rolling）吸收效率高。

2. ResNet1D：
	- 能顯式利用序列結構並透過殘差連線提高可訓練性。
	- 在本專案調校後，具備超越 MLP 的潛力與更高上限。

觀察與結論模板（每次實驗可複製填寫）：

1. 設定：
	- models=...
	- sequence_length=...
	- mlp_lr=..., resnet_lr=...
	- target_transform=...
	- early_stopping=... / min_epochs_before_stop=...

2. 指標：
	- Best Test RMSLE=...
	- Test R2=...
	- Test Peak Recall=...
	- Train time=...

3. 判讀：
	- 是否過擬合：觀察 Overfit Gap 與 Train/Test 指標差距。
	- 是否需要下一輪：
	  - 若 RMSLE 未改善：先調 sequence_length 或 lr。
	  - 若 Peak Recall 偏低：優先強化高峰日權重或特徵。