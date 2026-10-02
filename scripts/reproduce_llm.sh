#!/bin/bash
# Reproduce the emotion-conditioned LLM results in RESULTS.md on two GPUs:
#   emotion features -> pseudo labels -> SFT -> GRPO -> judge-scored eval -> held-out pairwise eval
# Each argument is <name>=<emotion checkpoint>; one policy is trained per emotion model, side by side.
#
#   POLICY_GPU=0 JUDGE_GPU=1 scripts/reproduce_llm.sh \
#       cls_attention=runs/finetune_<stamp>/best.pt supcon_attention=runs/finetune_<stamp>/best.pt
#
# Options (environment variables):
#   STAGES     subset of "features labels sft grpo eval pairwise" (default: all)
#   GRPO_TAG   name of the GRPO stage in runs/llm_index.tsv (default grpo; use a new tag to
#              compare reward settings, e.g. GRPO_TAG=grpo_cpmi COMPARE_TAG=grpo
#              EXTRA="grpo.reward.contrast_weight=0.5 grpo.reward.pmi_weight=0.1")
#   COMPARE_TAG  an earlier GRPO tag to compare against in the pairwise stage
#   SPLIT      split for the eval / pairwise stages (default dev; e.g. test)
#   EVAL_LIMIT utterances scored by the training judge in the eval stage (default 200; 0 = all)
#   EXTRA      overrides passed to SFT and GRPO
#   DATA_ROOT  (default data/meld)   VLLM  vLLM executable (default: vllm)   WANDB_MODE (default offline)
#
# The judge LLMs are served with vLLM on JUDGE_GPU (training judge for labels / GRPO / eval, then
# the held-out judge for the pairwise stage); policies train on POLICY_GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
POLICY_GPU=${POLICY_GPU:?set POLICY_GPU}; JUDGE_GPU=${JUDGE_GPU:?set JUDGE_GPU}
STAGES=${STAGES:-"features labels sft grpo eval pairwise"}
GRPO_TAG=${GRPO_TAG:-grpo}; COMPARE_TAG=${COMPARE_TAG:-}; EXTRA=${EXTRA:-}
SPLIT=${SPLIT:-dev}; EVAL_LIMIT=${EVAL_LIMIT:-200}
VLLM=${VLLM:-vllm}; ROOT=${DATA_ROOT:-data/meld}
COMMON=("data_root=$ROOT" "wandb.mode=${WANDB_MODE:-offline}")
LOG=runs/logs; INDEX=runs/llm_index.tsv; OUT=runs/llm_pairwise/$SPLIT
mkdir -p $LOG $OUT; [ -f $INDEX ] || printf "name\tstage\trun_dir\tstatus\n" > $INDEX
[ $# -gt 0 ] || { echo "usage: reproduce_llm.sh <name>=<emotion checkpoint> ..."; exit 1; }
declare -A CKPT; NAMES=()
for a in "$@"; do NAMES+=("${a%%=*}"); CKPT[${a%%=*}]=${a#*=}; done
# transcripts / dialogue context for label generation, baselines and judging (any model's cache)
TEXT=("data_root=$ROOT" "emotion.cache_dir=$ROOT/llm/${NAMES[0]}")

ts() { date +%H:%M:%S; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
rundir() { grep -oE "run dir runs/[^ ]+" "$1" | head -n 1 | cut -d' ' -f3; }
record() { printf "%s\t%s\t%s\t%s\n" "$1" "$2" "${3:--}" "$4" >> $INDEX; }
dir_of() { awk -F'\t' -v n="$1" -v s="$2" '$1 == n && $2 == s && $4 == "ok" {d = $3} END {print d}' $INDEX; }
policy() { CUDA_VISIBLE_DEVICES=$POLICY_GPU python "$@"; }

serve() {  # serve <model> <port> <gpu memory share> [extra vllm args]; waits until ready
  local model=$1 port=$2 mem=$3; shift 3
  if curl -s "localhost:$port/v1/models" | grep -q "$model"; then
    echo "[$(ts)] reusing $model already served on :$port"; return
  fi
  CUDA_VISIBLE_DEVICES=$JUDGE_GPU nohup $VLLM serve "$model" --port "$port" --gpu-memory-utilization "$mem" \
      --max-model-len 4096 --dtype bfloat16 --served-model-name "$model" "$@" > $LOG/vllm_$port.log 2>&1 &
  local pid=$!
  until curl -s "localhost:$port/v1/models" | grep -q "$model"; do
    kill -0 $pid 2>/dev/null || { echo "vLLM $model exited: $LOG/vllm_$port.log"; exit 1; }
    sleep 10
  done
  echo "[$(ts)] serving $model on :$port (GPU $JUDGE_GPU)"
}
unserve() { pkill -f "vllm serve $1" && sleep 20; true; }

# 1. emotion features per emotion model (train, dev and SPLIT; existing caches are kept)
if has features; then
  for n in "${NAMES[@]}"; do
    splits=$(for s in train dev $SPLIT; do [ -f "$ROOT/llm/$n/features_$s.pt" ] || echo $s; done | sort -u | paste -sd,)
    [ -z "$splits" ] && { echo "[$(ts)] features $n: cached"; continue; }
    policy scripts/llm/extract_features.py --splits $splits "${COMMON[@]}" emotion.checkpoint="${CKPT[$n]}" \
        emotion.cache_dir="$ROOT/llm/$n" > $LOG/llm_features_$n.log 2>&1 || { echo "features $n FAILED"; exit 1; }
    echo "[$(ts)] features $n ($splits): $(grep -E '^[a-z]+:' $LOG/llm_features_$n.log | cut -c1-110 | paste -sd' ')"
  done
fi

# 2. pseudo labels (text only, shared by every policy; cached and resumable)
needs_judge=0; for s in labels grpo eval; do has $s && needs_judge=1; done
[ $needs_judge = 1 ] && serve Qwen/Qwen2.5-32B-Instruct 8000 0.92
if has labels; then
  # the candidate model shares the judge GPU (the 32B judge leaves no room on it), so serve it on POLICY_GPU
  JUDGE_GPU_SAVED=$JUDGE_GPU; JUDGE_GPU=$POLICY_GPU; serve Qwen/Qwen2.5-3B-Instruct 8001 0.25; JUDGE_GPU=$JUDGE_GPU_SAVED
  python scripts/llm/make_pseudo_labels.py "${TEXT[@]}" > $LOG/llm_pseudo_labels.log 2>&1 \
      || { echo "pseudo labels FAILED: $LOG/llm_pseudo_labels.log"; exit 1; }
  grep -E "^(train|dev):" $LOG/llm_pseudo_labels.log
  unserve Qwen/Qwen2.5-3B-Instruct
fi

# 3. SFT, side by side (wait on the training jobs only, not on the vLLM servers)
declare -A SFT GRPO
jobs_=()
for n in "${NAMES[@]}"; do
  if has sft; then
    policy scripts/llm/train_sft.py "${COMMON[@]}" emotion.cache_dir="$ROOT/llm/$n" wandb.name=${n}_sft $EXTRA \
        > $LOG/llm_sft_$n.log 2>&1 &
    jobs_+=($!); sleep 60   # stagger model loading
  fi
done
[ ${#jobs_[@]} -gt 0 ] && wait "${jobs_[@]}"
for n in "${NAMES[@]}"; do
  if has sft; then
    d=$(rundir $LOG/llm_sft_$n.log)
    if [ -f "$d/policy.json" ]; then record $n sft $d ok; else record $n sft "$d" FAILED; fi
    echo "[$(ts)] SFT $n -> ${d:-FAILED} ($(grep -E '^policy' $LOG/llm_sft_$n.log | cut -c1-80))"
  fi
  SFT[$n]=$(dir_of $n sft)
done

# 4. GRPO, side by side (both share the judge)
jobs_=()
for n in "${NAMES[@]}"; do
  if has grpo && [ -f "${SFT[$n]}/policy.json" ]; then
    policy scripts/llm/train_grpo.py "${COMMON[@]}" emotion.cache_dir="$ROOT/llm/$n" grpo.init_from=${SFT[$n]} \
        wandb.name=${n}_$GRPO_TAG $EXTRA > $LOG/llm_${GRPO_TAG}_$n.log 2>&1 &
    jobs_+=($!); sleep 90
  fi
done
[ ${#jobs_[@]} -gt 0 ] && wait "${jobs_[@]}"
for n in "${NAMES[@]}"; do
  if has grpo; then
    d=$(rundir $LOG/llm_${GRPO_TAG}_$n.log)
    if [ -f "$d/policy.json" ]; then record $n $GRPO_TAG $d ok; else record $n $GRPO_TAG "$d" FAILED; fi
    echo "[$(ts)] GRPO ($GRPO_TAG) $n -> ${d:-FAILED}"
  fi
  GRPO[$n]=$(dir_of $n $GRPO_TAG)
done

# 5. training-judge scores on SPLIT (EVAL_LIMIT utterances; 0 = all)
if has eval; then
  for n in "${NAMES[@]}"; do
    for d in "${SFT[$n]}" "${GRPO[$n]}"; do
      [ -f "$d/policy.json" ] || continue
      policy scripts/llm/evaluate.py "$d" --split $SPLIT --limit $EVAL_LIMIT "data_root=$ROOT" \
          > $LOG/llm_eval_${SPLIT}_$(basename $d).log 2>&1 &&
        echo "[$(ts)] eval $n $(basename $d): $(grep '^{' $LOG/llm_eval_${SPLIT}_$(basename $d).log | cut -c1-160)"
    done
  done
  policy scripts/llm/evaluate.py baseline:none --split $SPLIT --limit $EVAL_LIMIT "${TEXT[@]}" \
      > $LOG/llm_eval_${SPLIT}_baseline_none.log 2>&1 &&
    echo "[$(ts)] eval no-emotion baseline: $(grep '^{' $LOG/llm_eval_${SPLIT}_baseline_none.log | cut -c1-160)"
fi
unserve Qwen/Qwen2.5-32B-Instruct

# 6. held-out pairwise judging + emotion alignment on all of SPLIT (replies are generated once
#    per system and split, and cached; see emollm/systems.py)
if has pairwise; then
  serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 8002 0.92 \
      --tokenizer-mode mistral --config-format mistral --load-format mistral
  pair() {  # pair <name A> <system A> <name B> <system B> [evaluate_pairwise.py args]
    local na=$1 a=$2 nb=$3 b=$4; shift 4
    local tag=${na}_vs_${nb}${1:+_$(echo "$*" | tr -d ' -')}
    policy scripts/llm/evaluate_pairwise.py "$a" "$b" --split $SPLIT --out $OUT/$tag.json "$@" "data_root=$ROOT" \
        > $LOG/llm_pairwise_${SPLIT}_$tag.log 2>&1 || { echo "pairwise $tag FAILED"; return; }
    echo "[$(ts)] $tag"; grep -E "label=|alignment gap" $LOG/llm_pairwise_${SPLIT}_$tag.log
  }
  for n in "${NAMES[@]}"; do
    g=${GRPO[$n]}; [ -f "$g/policy.json" ] || continue
    pair ${n}_$GRPO_TAG "$g" no_emotion baseline:none
    pair ${n}_$GRPO_TAG "$g" no_emotion baseline:none --counterfactual valence
    pair ${n}_$GRPO_TAG "$g" ${n}_${GRPO_TAG}_shuffled "$g:shuffled"
    pair ${n}_$GRPO_TAG "$g" ${n}_sft "${SFT[$n]}"
    [ -n "$COMPARE_TAG" ] && pair ${n}_$GRPO_TAG "$g" ${n}_$COMPARE_TAG "$(dir_of $n $COMPARE_TAG)"
  done
  [ ${#NAMES[@]} -ge 2 ] && pair ${NAMES[0]}_$GRPO_TAG "${GRPO[${NAMES[0]}]}" ${NAMES[1]}_$GRPO_TAG "${GRPO[${NAMES[1]}]}"
  unserve mistralai/Mistral-Small-3.2-24B-Instruct-2506
fi
echo "[$(ts)] done: $INDEX, $OUT"
