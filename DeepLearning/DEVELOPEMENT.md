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