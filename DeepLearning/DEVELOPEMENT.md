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

若要關閉 target transform：

```bash
python main.py --models mlp,resnet1d --skip-plot --target-transform none
```