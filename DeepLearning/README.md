大家好:

請大家下載附件 Deep Learning Midterm Programming 202604.zip 檔案，

壓縮檔中的 Midterm Programming 202604.docx 文件

為本次期中考程式的題目，並填寫文件中的表單。

請統一使用 Root Mean Squared Log Error (RMSLE) 作為評估方法。


將表單與程式碼一同壓縮並繳交至此 (壓縮檔需包含所有用到的文件 如：XXX.csv...等)
請將壓縮檔取名為「學號」，範例：「614XXXXXX」。

繳交期限:  2026/05/07 (deadline: 23:59:59)   50% of total grade of mid-term exam (不可逾期繳交)

補充：

1. cost 和 loss function 寫同一個東西即可

2. 如果使用的不是 Root Mean Squared Log Error (RMSLE) 作為評估方法。則需簡單提及你的評估方式。

3. 如果使用的評估指標不是用 % 為單位則輸出 Word 檔案時請把它去掉。

---

目前程式評估設定（對齊題目）：

1. 主評估指標為 Root Mean Squared Log Error (RMSLE)。
2. 測試集評分會排除 visitors = 0 的樣本（closed days）。
3. 輸出結果 `results.csv` 使用 `Train_RMSLE` 與 `Test_RMSLE` 欄位，不使用百分比 accuracy。

程式碼結構說明：

1. 主要執行入口為 `main.py`。
2. 正式實作統一放在 `src/` 底下（`src/data`、`src/preprocessing`、`src/models`、`src/evaluation`）。

執行環境資訊：

1. Python 版本：3.11.15
2. conda 環境名稱：dl-class
3. 套件版本請參考 requirements.txt

日誌輸出：

1. 使用 loguru 輸出訓練流程日誌。
2. 終端機顯示 INFO 以上訊息。
3. 同步寫入 logs/training.log（自動 rotation）。

預設訓練模型：

1. 目前直接執行 `python main.py` 時，預設只會訓練 `mlp,resnet1d`。
2. 其他模型（如 lgbm/lstm/cnn1d/transformer）需透過 `--models` 參數手動指定。

近期改善作法與觀察：

1. 特徵工程：新增每店鋪 `visitors_lag_1/7/14` 與 `visitors_roll_mean_7/visitors_roll_std_7`。
2. 模型穩定化：將 MLP/ResNet1D 輸出改為 Softplus，避免 ReLU 輸出層造成梯度截斷與全零預測。
3. ResNet1D 優化：降低預設 learning rate（3e-4）並調整 dropout，改善訓練崩壞問題。
4. 訓練策略：神經網路支援 validation split、early stopping 與 `log1p` target transform。

目前觀察（2026-04-16）：

1. ResNet1D 在新特徵與穩定化設定下可大幅優於舊版設定。
2. MLP 仍是穩定 baseline，適合與 ResNet1D 做快速 A/B 比較。
3. 下一步建議優先搜尋 `sequence_length` 與 early stopping patience，再調整模型深度。

可調參數（神經網路）：

1. `--mlp-lr`：MLP learning rate（預設 `1e-3`）。
2. `--resnet-lr`：ResNet1D learning rate（預設 `3e-4`）。
3. `--disable-early-stopping`：完全關閉 early stopping。
4. `--min-epochs-before-stop`：設定 early stopping 最少啟動 epoch。

自動化搜尋：

1. 可直接使用 [scripts/grid_search_mlp_resnet.sh](scripts/grid_search_mlp_resnet.sh) 進行 `sequence_length x lr` 搜尋。
2. 結果會輸出到 `grid_search_runs/grid_summary.csv`，並保留每次 run 的 `results.csv` 與 `best_model_summary.md`。

模型取捨原因（為何聚焦 MLP 與 ResNet1D）：

1. 傳統模型中僅保留 LGBM 做 baseline：可作為非深度學習基準，訓練速度快且穩定。
2. 線性模型/隨機森林/部分 boosting 在本資料設定下測試表現明顯落後，且額外執行時間無法換取更好 RMSLE。
3. MLP 作為穩定強 baseline：
	- 對 tabular+時間衍生特徵通常表現穩定。
	- 訓練時間較短，適合做大量參數比較。
4. ResNet1D 作為主力序列模型：
	- 殘差連線可改善深層網路訓練穩定性。
	- 在加入 lag/rolling 特徵與輸出層穩定化後，測試指標可明顯提升。

調校策略（方法與原因）：

1. 先修資料再修模型：先補齊時間軸與特徵縮放，再做模型深度優化，避免把資料問題誤判成模型問題。
2. 先建立穩定 baseline 再擴展：先跑 MLP，確認資料與評估流程正常，再導入 ResNet1D。
3. 以可解釋小步驟迭代：每次只改 1~2 類變數（例如輸出激活或 learning rate），保留對照可追蹤性。
4. 以 RMSLE 作主優化目標：配合 log1p target transform，讓訓練空間更對齊評分指標。

已記錄的關鍵觀察：

1. 輸出層激活函數影響很大：
	- ReLU 輸出層曾造成預測偏零與 Peak Recall 偏低。
	- 改為 Softplus 後，梯度更平滑且泛化更穩定。
2. 時間特徵工程有效：
	- 加入 lag/rolling 後，RMSLE 與 R2 同步改善。
3. ResNet1D 需要較保守 learning rate：
	- `3e-4` 相比 `1e-3` 更不容易出現訓練崩壞。
4. 早停需謹慎：
	- 過早觸發可能錯過後段收斂。
	- 已提供可關閉 early stopping 與 minimum epoch 門檻參數。

實驗紀錄建議：

1. 每次訓練前先備份 `results.csv` 與 `best_model_summary.md`，避免 smoke test 覆蓋正式結果。
2. 記錄至少以下欄位：
	- 模型組合、sequence_length、learning rate、是否 early stopping、target transform。
	- Test RMSLE、Test R2、Test Peak Recall、訓練時間。