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