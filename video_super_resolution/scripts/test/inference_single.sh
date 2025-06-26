#!/usr/bin/env bash
# inference_single.sh
# --------------------
# Usage:
#   ./inference_single.sh \
#     --input ./input/video.mp4 \
#     --prompt "my prompt" \
#     --model_path /path/to/light_deg.pt \
#     --save_dir ./results \
#     [--solver_mode fast] [--steps 15] [--cfg 7.5] [--upscale 4] [--max_chunk_len 32] \
#     [--log_dir ./logs] [--interval 60]
#
# This will:
#   • name the log      <base>_<timestamp>.log
#   • name the output   <base>_<timestamp>.mp4
#   • inside the log, you get header + inference stdout + metrics.

set -euo pipefail

####################
# Default settings
####################
INTERVAL=60
LOG_DIR="./logs"

SOLVER_MODE="fast"
STEPS=15
CFG=7.5
UPSCALE=4
MAX_CHUNK_LEN=32

####################
# Help & arg parsing
####################
usage() {
  grep '^#' "$0" | sed 's/^#//'
  exit 1
}

while [[ "${1-}" =~ ^- ]]; do
  case "$1" in
    --input)         VIDEO="$2"; shift 2;;
    --prompt)        PROMPT="$2"; shift 2;;
    --model_path)    MODEL_PATH="$2"; shift 2;;
    --save_dir)      SAVE_DIR="$2"; shift 2;;
    --solver_mode)   SOLVER_MODE="$2"; shift 2;;
    --steps)         STEPS="$2"; shift 2;;
    --cfg)           CFG="$2"; shift 2;;
    --upscale)       UPSCALE="$2"; shift 2;;
    --max_chunk_len) MAX_CHUNK_LEN="$2"; shift 2;;
    --log_dir)       LOG_DIR="$2"; shift 2;;
    --interval)      INTERVAL="$2"; shift 2;;
    --help)          usage;;
    *)               echo "Unknown flag $1" >&2; usage;;
  esac
done

# Validate required args
if [[ -z "${VIDEO-}" || -z "${MODEL_PATH-}" || -z "${SAVE_DIR-}" ]]; then
  echo "Error: --input, --model_path and --save_dir are required." >&2
  usage
fi

# Prepare directories
mkdir -p "$LOG_DIR" "$SAVE_DIR"

# Base name of the video (no extension)
BASE=$(basename "$VIDEO" .mp4)

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Filenames
RUN_BASENAME="${BASE}_${TIMESTAMP}"
LOGFILE="${LOG_DIR}/${RUN_BASENAME}.log"
OUTPUT_VIDEO="${RUN_BASENAME}.mp4"

####################
# 1) Write header into log
####################
{
  echo "Video   : $VIDEO"
  echo "Prompt  : ${PROMPT:-<none>}"
  echo "Settings: solver=${SOLVER_MODE}, steps=${STEPS}, cfg=${CFG}, upscale=${UPSCALE}, chunk_len=${MAX_CHUNK_LEN}"
  echo "Output  : ${SAVE_DIR}/${OUTPUT_VIDEO}"
  echo
} > "$LOGFILE"

####################
# 2) Launch inference.py
####################
python ./video_super_resolution/scripts/inference.py \
  --input_path "$VIDEO" \
  --file_name "$OUTPUT_VIDEO" \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  ${PROMPT:+--prompt "$PROMPT"} \
  --solver_mode "$SOLVER_MODE" \
  --steps "$STEPS" \
  --cfg "$CFG" \
  --upscale "$UPSCALE" \
  --max_chunk_len "$MAX_CHUNK_LEN" \
  2>&1 >> "$LOGFILE" &

PID=$!

####################
# 3) Start metrics logger
####################
source ./video_super_resolution/scripts/logger.sh
start_metrics_logger "$PID" "$INTERVAL" "$LOGFILE" &

# Wait for inference to finish
wait "$PID"

echo "Inference done for $BASE"
echo "Log      → $LOGFILE"
echo "Video    → ${SAVE_DIR}/${OUTPUT_VIDEO}"
