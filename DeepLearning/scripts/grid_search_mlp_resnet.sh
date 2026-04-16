#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/grid_search_mlp_resnet.sh
# Optional env vars:
#   SEQ_LIST="7 14 21"
#   MLP_LR_LIST="0.001 0.002"
#   RESNET_LR_LIST="0.0002 0.0003"
#   MLP_EPOCHS=100
#   RESNET_EPOCHS=100
#   EXTRA_ARGS="--target-transform log1p --disable-early-stopping"

SEQ_LIST=${SEQ_LIST:-"7 14 21 28"}
MLP_LR_LIST=${MLP_LR_LIST:-"0.0005 0.001 0.002"}
RESNET_LR_LIST=${RESNET_LR_LIST:-"0.0001 0.0002 0.0003 0.0005"}
MLP_EPOCHS=${MLP_EPOCHS:-100}
RESNET_EPOCHS=${RESNET_EPOCHS:-100}
EXTRA_ARGS=${EXTRA_ARGS:-"--target-transform log1p --val-start-date 2016-10-01 --disable-early-stopping"}

mkdir -p grid_search_runs
summary_file="grid_search_runs/grid_summary.csv"

# header
printf "run_id,model,test_rmsle,test_rmse,test_mae,test_r2,test_peak_recall,sequence_length,mlp_lr,resnet_lr,mlp_epochs,resnet_epochs\n" > "$summary_file"

run_id=0
for seq in $SEQ_LIST; do
  for mlp_lr in $MLP_LR_LIST; do
    for resnet_lr in $RESNET_LR_LIST; do
      run_id=$((run_id + 1))
      echo "=== Grid Run ${run_id}: seq=${seq}, mlp_lr=${mlp_lr}, resnet_lr=${resnet_lr} ==="

      python main.py \
        --models mlp,resnet1d \
        --sequence-length "$seq" \
        --mlp-epochs "$MLP_EPOCHS" \
        --resnet-epochs "$RESNET_EPOCHS" \
        --mlp-lr "$mlp_lr" \
        --resnet-lr "$resnet_lr" \
        --skip-plot \
        $EXTRA_ARGS

      run_dir="grid_search_runs/run_${run_id}_seq${seq}_mlp${mlp_lr}_resnet${resnet_lr}"
      mkdir -p "$run_dir"
      cp results.csv "$run_dir/results.csv"
      cp best_model_summary.md "$run_dir/best_model_summary.md"

      # Append each model row with hyperparameters
      awk -F',' -v OFS=',' -v run_id="$run_id" -v seq="$seq" -v mlp_lr="$mlp_lr" -v resnet_lr="$resnet_lr" -v me="$MLP_EPOCHS" -v re="$RESNET_EPOCHS" '
        NR==1 {next}
        {
          # columns: Model,...,Test_RMSLE (18), ... Test_RMSE(12), Test_MAE(13), Test_R2(14), Test_Peak_Recall(19)
          print run_id, $1, $18, $12, $13, $14, $19, seq, mlp_lr, resnet_lr, me, re
        }
      ' results.csv >> "$summary_file"
    done
  done
done

echo "\nGrid search complete. Raw summary: $summary_file"

echo "Top 10 by test_rmsle:"
{
  head -n 1 "$summary_file"
  tail -n +2 "$summary_file" | sort -t',' -k3,3g | head -n 10
} | column -s',' -t
