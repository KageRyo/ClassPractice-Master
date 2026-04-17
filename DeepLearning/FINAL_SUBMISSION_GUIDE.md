# Final Submission Guide

## 1) Locked Final Command

```bash
conda activate dl-class
python main.py
```

`python main.py` defaults are already aligned with the final configuration:
- models: `mlp,resnet1d`
- epochs: 100/100
- sequence length: 14
- learning rate: `mlp=0.001`, `resnet1d=0.0002`
- target transform: `log1p`
- validation split start: `2016-10-01`
- plotting: skipped by default

## 2) Final Outputs to Submit

- `results.csv`
- `best_model_summary.md`
- source code in this repository

## 3) Locked Final Metrics

- Best Model: `resnet1d`
- Train RMSLE: `0.558909`
- Test RMSLE: `0.560029`
- Train R2: `0.633555`
- Test R2: `-0.973978`
- Overfit Gap: `0.001120` (`No`)

## 4) Pre-Submission Checklist

- Confirm `results.csv` contains both `mlp` and `resnet1d`
- Confirm `best_model_summary.md` matches the locked final run
- Confirm report text in `Mid-Term Programming Exam.md` is filled
- Confirm README and development notes are consistent with the code
