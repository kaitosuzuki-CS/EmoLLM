#!/bin/bash
# Reproduce every emotion-model result in RESULTS.md on one GPU, one run at a time:
#   data preparation (skipped if done) -> 17 experiments (model selection on dev) ->
#   test evaluation of every checkpoint -> kNN baseline -> latent-space plots (train / dev / test)
#
#   CUDA_VISIBLE_DEVICES=0 scripts/reproduce_emotion.sh            # everything
#   CUDA_VISIBLE_DEVICES=0 scripts/reproduce_emotion.sh sweep_     # only experiments named sweep_*
#   DATA_ROOT=/data/meld CUDA_VISIBLE_DEVICES=0 scripts/reproduce_emotion.sh
#
# runs/rerun_index.tsv maps experiment names to run directories; logs go to runs/logs/;
# latent plots to runs/latent/<experiment>/.
set -uo pipefail
cd "$(dirname "$0")/.."
FILTER=${1:-}
DATA=("data_root=${DATA_ROOT:-data/meld}")
W="wandb.mode=${WANDB_MODE:-offline}"
mkdir -p runs/logs runs/latent
INDEX=runs/rerun_index.tsv
[ -f $INDEX ] || printf "name\trun_dir\tstatus\tcommand\n" > $INDEX
ts() { date +%H:%M:%S; }

# 1. data: 16 kHz wavs + manifests, then cached emotion2vec+ z and frame features
cache=$(python -c "from emotion.config import load_config as l; print(l('configs/emotion/base.yaml', ['${DATA[0]}'])['data']['splits']['test']['frame_cache'])")
if [ ! -f "$cache" ]; then
  echo "[$(ts)] preparing data"
  python scripts/data/prepare_audio.py --config configs/emotion/base.yaml --splits train,dev,test "${DATA[@]}" \
      > runs/logs/prepare_audio.log 2>&1 || exit 1
  python scripts/data/extract_embeddings.py --config configs/emotion/base.yaml --splits train,dev,test --frames \
      "${DATA[@]}" > runs/logs/extract_embeddings.log 2>&1 || exit 1
fi

run() {  # run <name> <script> <args...>
  local name=$1; shift
  [[ -n $FILTER && $name != $FILTER* ]] && return
  local log=runs/logs/$name.log
  echo "[$(ts)] $name"
  python "$@" "${DATA[@]}" $W > "$log" 2>&1
  local status=$? dir
  dir=$(grep -oE "run dir runs/[^ ]+" "$log" | head -n 1 | cut -d' ' -f3)
  printf "%s\t%s\t%s\t%s\n" "$name" "${dir:--}" "$([ $status = 0 ] && echo ok || echo "FAILED($status)")" \
      "python $*" >> $INDEX
}

# 2. main table: 4 pipelines x {mean, attention} pooling
for p in mean attention; do
  run main_frozen_cls_$p        scripts/emotion/train_classifier.py --config configs/emotion/classification.yaml pooling.type=$p
  run main_frozen_supcon_$p     scripts/emotion/train_supcon.py     --config configs/emotion/supcon.yaml         pooling.type=$p
  run main_finetune_cls_$p      scripts/emotion/train_finetune.py   --config configs/emotion/finetune.yaml       pooling.type=$p
  run main_finetune_supcon_$p   scripts/emotion/train_finetune.py   --config configs/emotion/finetune_supcon.yaml pooling.type=$p
done

# 3. ablations
run ablation_frozen_supcon_plain_100ep scripts/emotion/train_supcon.py --config configs/emotion/supcon.yaml \
    loss.var_weight=0 loss.cov_weight=0
for vc in "0 0" "1 0.04" "1 0.2" "5 0.2" "25 1"; do set -- $vc
  run "sweep_var$1_cov$2" scripts/emotion/train_supcon.py --config configs/emotion/supcon.yaml train.epochs=30 \
      loss.var_weight=$1 loss.cov_weight=$2
done
run ablation_finetune_supcon_plain_3ep scripts/emotion/train_finetune.py --config configs/emotion/finetune_supcon.yaml \
    train.epochs=3 loss.var_weight=0 loss.cov_weight=0
run ablation_finetune_supcon_varcov_2ep scripts/emotion/train_finetune.py --config configs/emotion/finetune_supcon.yaml \
    train.epochs=2 train.freeze_encoder_epochs=0
run ablation_finetune_supcon_varcov_1ep scripts/emotion/train_finetune.py --config configs/emotion/finetune_supcon.yaml \
    train.epochs=1 train.freeze_encoder_epochs=0

# 4. test-set evaluation of every checkpoint (evaluation only: epochs were selected on dev)
tail -n +2 $INDEX | while IFS=$'\t' read -r name dir status _; do
  [[ -n $FILTER && $name != $FILTER* ]] && continue
  [ "$status" = ok ] || continue
  python scripts/emotion/evaluate.py "$dir" --split test --fit-probe --batch-size 32 \
      --data-root "${DATA_ROOT:-data/meld}" \
      > runs/logs/test_eval_$name.log 2>&1 || echo "  test eval FAILED: runs/logs/test_eval_$name.log"
done

# 5. no-training baseline (dev and test)
if [[ -z $FILTER ]]; then
  python scripts/emotion/knn_baseline.py --split dev "${DATA[@]}" | tee runs/logs/knn_baseline_dev.log
  python scripts/emotion/knn_baseline.py --split test "${DATA[@]}" | tee runs/logs/knn_baseline_test.log
fi

# 6. latent-space plots of every checkpoint (the untrained source "raw" is plotted once)
first=1
tail -n +2 $INDEX | while IFS=$'\t' read -r name dir status _; do
  [[ -n $FILTER && $name != $FILTER* ]] && continue
  [ "$status" = ok ] || continue
  sources=z,h; [ $first = 1 ] && sources=raw,z,h; first=0
  echo "[$(ts)] latent plots: $name"
  python scripts/emotion/plot_latent_space.py "$dir" --sources $sources --out runs/latent/$name --name $name \
      --data-root "${DATA_ROOT:-data/meld}" \
      > runs/logs/latent_$name.log 2>&1 || echo "  FAILED: runs/logs/latent_$name.log"
done
echo "[$(ts)] done: $INDEX"
