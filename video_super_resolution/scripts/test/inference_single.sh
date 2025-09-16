#!/usr/bin/env bash
# inference_single.sh
# --------------------
# Usage:
#   ./inference_single.sh \
#     --input ./input/video.mp4 \
#     [--prompt "my prompt"] \
#     [--model_path /path/to/light_deg.pt] \
#     [--save_dir ./results] \
#     [--solver_mode fast] [--steps 15] [--cfg 7.5] [--upscale 4] [--max_chunk_len 32] \
#     [--log_dir ./logs] [--interval 60]
#
# This will:
#   • name the log    <basename>_<timestamp>.log
#   • name the output <basename>_<timestamp>.mp4
#   • inside the log: header (including input size), inference stdout, then metrics.

set -euo pipefail

####################
# Defaults
####################
INTERVAL=60
LOG_DIR="./logs"

MODEL_PATH="/opt/openhardware/pretrained_weights/light_deg.pt"
SAVE_DIR="./results"

SOLVER_MODE="fast"
STEPS=15
CFG=7.5
UPSCALE=2
MAX_CHUNK_LEN=12

####################
# Arg parsing
####################
usage() {
  grep '^#' "$0" | sed 's/^#//'
  exit 1
}

# Temporary prompt variable (can be overridden)
PROMPT=""

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

if [[ -z "${VIDEO-}" ]]; then
  echo "Error: --input is required." >&2
  usage
fi

# If no prompt passed via flag, read first non-empty line from file
PROMPT_FILE="./input/text/prompt.txt"
if [[ -z "$PROMPT" && -f "$PROMPT_FILE" ]]; then
  PROMPT=$(grep -v '^[[:space:]]*$' "$PROMPT_FILE" | head -n 1)
fi

mkdir -p "$LOG_DIR" "$SAVE_DIR"

BASE=$(basename "$VIDEO" .mp4)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_BASENAME="${BASE}_${TIMESTAMP}"
LOGFILE="${LOG_DIR}/${RUN_BASENAME}.log"
OUTPUT_VIDEO="${RUN_BASENAME}.mp4"

# Probe input size
if command -v ffprobe &>/dev/null; then
  read WIDTH HEIGHT < <(ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=width,height \
    -of csv=p=0 "$VIDEO")
  FPS_RAW=$(ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$VIDEO")
  DURATION=$(ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$VIDEO")
  FRAME_COUNT=$(python - "$FPS_RAW" "$DURATION" <<'PYCODE'
import sys, fractions
fps = fractions.Fraction(sys.argv[1])
dur = float(sys.argv[2])
print(int(round(dur * fps)))
PYCODE
  )
else
  WIDTH="?"
  HEIGHT="?"
  FPS_RAW="?"
  FRAME_COUNT="?"
fi

# Write header into log
{
  echo "Video       : $VIDEO"
  echo "Resolution  : ${WIDTH}x${HEIGHT}"
  echo "FPS         : $FPS_RAW"
  echo "Frame count : $FRAME_COUNT"
  echo "Prompt      : ${PROMPT:-<none>}"
  echo "Settings    : solver=${SOLVER_MODE}, steps=${STEPS}, cfg=${CFG}, upscale=${UPSCALE}, chunk_len=${MAX_CHUNK_LEN}"
  echo "Model       : ${MODEL_PATH}"
  echo "Output file : ${SAVE_DIR}/${OUTPUT_VIDEO}"
  echo
} > "$LOGFILE"

# Run inference
python ./video_super_resolution/scripts/inference.py \
  --input_path "$VIDEO" \
  --file_name "$OUTPUT_VIDEO" \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --prompt "$PROMPT" \
  --solver_mode "$SOLVER_MODE" \
  --steps "$STEPS" \
  --cfg "$CFG" \
  --upscale "$UPSCALE" \
  --max_chunk_len "$MAX_CHUNK_LEN" \
  2>&1 >> "$LOGFILE" &

PID=$!

# Start metrics logger
source ./video_super_resolution/scripts/logger.sh
start_metrics_logger "$PID" "$INTERVAL" "$LOGFILE" &

wait "$PID"

echo "Inference done for $BASE"
echo "Log   → $LOGFILE"
echo "Video → ${SAVE_DIR}/${OUTPUT_VIDEO}"
echo "Log   → $LOGFILE"
echo "Video → ${SAVE_DIR}/${OUTPUT_VIDEO}"
