#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dust_env311

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/mutiDRL}
cd "$PROJECT_DIR"

mkdir -p logs runs
RUN_DIR="runs/$(date +%Y%m%d_%H%M%S)"
TRAIN_DIR="$RUN_DIR/train"
DQN_DIR="$RUN_DIR/dqn"
BASELINE_DIR="$RUN_DIR/baseline"
COMPARE_DIR="$RUN_DIR/compare"
mkdir -p "$TRAIN_DIR" "$DQN_DIR" "$BASELINE_DIR" "$COMPARE_DIR"

echo "$RUN_DIR" > logs/latest_run_dir.txt

export SMART_SITE_HEADLESS=1
export MPLCONFIGDIR="$RUN_DIR/matplotlib"
SEED="${SMART_SITE_SEED:-20260514}"

collect_dataset() {
  local out_dir="$1"
  local need_rows="${2:-3000}"

  mkdir -p "$out_dir"
  rm -f "$out_dir/dust_dataset.csv" "$out_dir/cannon_command.txt" "$out_dir/latest_frame.jpg"

  SMART_SITE_OUTPUT_DIR="$out_dir" SMART_SITE_SEED="$SEED" python collector.py &
  local pid=$!

  while [ ! -f "$out_dir/dust_dataset.csv" ] || [ "$(($(wc -l < "$out_dir/dust_dataset.csv")-1))" -lt "$need_rows" ]; do
    rows=0
    [ -f "$out_dir/dust_dataset.csv" ] && rows=$(($(wc -l < "$out_dir/dust_dataset.csv")-1))
    echo "collecting rows in $out_dir: $rows/$need_rows"

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "collector exited unexpectedly"
      exit 1
    fi

    sleep 10
  done

  echo "initial data collected in $out_dir, stopping collector"
  kill "$pid"
  wait "$pid" 2>/dev/null || true
}

echo "========== Run dir: $RUN_DIR =========="

echo "========== Step 1: collect training data, 3000 rows =========="
collect_dataset "$TRAIN_DIR" 3000

echo "========== Step 2: train LSTM, BiLSTM, Attention-LSTM; seq_len=40, horizon=5, max 100 epochs =========="
SMART_SITE_OUTPUT_DIR="$TRAIN_DIR" \
SMART_SITE_SEQ_LEN=40 \
SMART_SITE_PREDICT_HORIZON_STEPS=5 \
SMART_SITE_PREDICT_EPOCHS=100 \
python 预测1.py

echo "========== Step 3: run DQN controller =========="
cp "$TRAIN_DIR/prediction_preprocessor.pkl" "$DQN_DIR/prediction_preprocessor.pkl"

SMART_SITE_OUTPUT_DIR="$DQN_DIR" \
SMART_SITE_RUNTIME_PREPARED=1 \
SMART_SITE_DQN_MODEL_FILE="$TRAIN_DIR/dust_attention_lstm_model.keras" \
SMART_SITE_DQN_POLICY_MODEL_FILE="$DQN_DIR/dqn_policy_model.keras" \
python drl_controller.py --max-control-steps 5000 --seed "$SEED"

echo "========== Step 4: run threshold baseline =========="
SMART_SITE_OUTPUT_DIR="$BASELINE_DIR" \
SMART_SITE_RUNTIME_PREPARED=1 \
python threshold_baseline_controller.py --max-control-steps 5000 --seed "$SEED"

echo "========== Step 5: compare control experiments =========="
python compare_control_experiments.py \
  --log DQN="$DQN_DIR/dqn_control_log.csv" \
  --log Threshold="$BASELINE_DIR/threshold_baseline_control_log.csv" \
  --output-dir "$COMPARE_DIR"

echo "========== all done =========="
echo "Results saved in: $RUN_DIR"
