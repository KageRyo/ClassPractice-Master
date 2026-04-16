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