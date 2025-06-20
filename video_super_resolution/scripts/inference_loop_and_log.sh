#!/usr/bin/env bash
# run_and_log.sh
# wrapper that invokes inference_loop.sh and logger.sh

# defaults
INTERVAL=60
LOGDIR="./logs"
SCRIPT="video_super_resolution/scripts/inference_loop.sh"

# parse args
while [[ "$1" =~ ^- ]]; do
  case "$1" in
    --interval)
      INTERVAL="$2"; shift 2;;
    --log-dir|-d)
      LOGDIR="$2"; shift 2;;
    --)
      shift; break;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# make sure log directory exists
mkdir -p "$LOGDIR"

# remaining args for inference_loop.sh
INFER_ARGS=("$@")

# launch inference
"$SCRIPT" "${INFER_ARGS[@]}" &
PID=$!
echo "Launched inference loop (PID $PID)"

# prepare logfile name under chosen folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR%/}/metrics_${TIMESTAMP}.log"

# start logger
source video_super_resolution/scripts/logger.sh
start_metrics_logger "$PID" "$INTERVAL" "$LOGFILE"
