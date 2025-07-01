#!/usr/bin/env bash
# inference_batch.sh
# ------------------
# Loops over all MP4s in a folder, reads prompts one-per-line,
# and invokes inference_single.sh on each video.

set -euo pipefail

VIDEO_DIR='./input/video'
PROMPT_FILE='./input/text/prompt.txt'
FRAME_LEN=24
SCRIPT_SINGLE=./video_super_resolution/scripts/test/inference_single.sh

mapfile -t VIDEOS < <(find "$VIDEO_DIR" -type f -name '*.mp4' | sort)
mapfile -t PROMPTS < <(grep -v '^\s*$' "$PROMPT_FILE")

if [ "${#VIDEOS[@]}" -ne "${#PROMPTS[@]}" ]; then
  echo "Error: ${#VIDEOS[@]} videos vs ${#PROMPTS[@]} prompts—counts must match."
  exit 1
fi

echo "Starting batch of ${#VIDEOS[@]} videos..."

for i in "${!VIDEOS[@]}"; do
  V="${VIDEOS[$i]}"
  P="${PROMPTS[$i]}"

  echo "Processing [$((i+1))/${#VIDEOS[@]}]: $V"

  "$SCRIPT_SINGLE" \
    --input "$V" \
    --prompt "$P" \
    --model_path /mnt/hdd3/openhardware/star/pretrained_weights/light_deg.pt \
    --save_dir ./results \
    --solver_mode fast \
    --steps 15 \
    --cfg 7.5 \
    --upscale 4 \
    --max_chunk_len "$FRAME_LEN" \
    --log_dir ./logs \
    --interval 60

  echo "Finished $V"
done

echo "Batch complete. Logs are in ./logs"
