#!/usr/bin/env bash
# inference_batch.sh
# ------------------
# Loops over all MP4s in a folder, reads a matching prompt for each
# from a text file, and invokes inference_single.sh on each one.

set -euo pipefail

VIDEO_DIR='./input/video'
PROMPT_FILE='./input/text/prompt.txt'
FRAME_LEN=32       # max_chunk_len default
SCRIPT_SINGLE=./video_super_resolution/scripts/test/inference_single.sh

# Read .mp4 files (in alphanumeric order)
mapfile -t VIDEOS < <(find "$VIDEO_DIR" -type f -name '*.mp4' | sort)

# Read prompts (one per line, skip blanks)
mapfile -t PROMPTS < <(grep -v '^\s*$' "$PROMPT_FILE")

if [ "${#VIDEOS[@]}" -ne "${#PROMPTS[@]}" ]; then
  echo "${#VIDEOS[@]} videos vs ${#PROMPTS[@]} prompts—counts must match."
  exit 1
fi

echo "Found ${#VIDEOS[@]} videos; starting batch inference..."

for i in "${!VIDEOS[@]}"; do
  V="${VIDEOS[$i]}"
  P="${PROMPTS[$i]}"
  echo "---"
  echo "▶︎ [$((i+1))/${#VIDEOS[@]}] $V"
  
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

  echo "Done $V"
done

echo "Batch complete; all logs are in ./logs"
