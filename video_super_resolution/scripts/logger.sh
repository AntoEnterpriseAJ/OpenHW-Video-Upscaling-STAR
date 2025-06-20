#!/usr/bin/env bash
# video_super_resolution/scripts/logger.sh
# Library to log CPU, RAM, GPU metrics for a given PID every INTERVAL seconds.
# Usage: source logger.sh && start_metrics_logger <PID> [INTERVAL] <LOGFILE>

start_metrics_logger() {
  local PID="$1"
  local INTERVAL="${2:-60}"
  local LOGFILE="$3"

  # Fetch system totals once
  local TOTAL_MEM_KB
  TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  local TOTAL_VRAM_MB
  TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total \
      --format=csv,noheader,nounits -i 0 | tr -d ' ')

  # Write header
  {
    printf "%-19s | %6s | %8s | %6s | %6s | %8s | %6s | %s\n" \
      "Timestamp" "CPU%" "RAM(MB)" "RAM%" "GPU0%" "VRAM(MB)" "VRAM%" "Per-Core(%)"
    printf '%.0s─' {1..120}
    echo
  } > "$LOGFILE"

  # Sampling loop
  while kill -0 "$PID" 2>/dev/null; do
    TS=$(date +%Y-%m-%dT%H:%M:%S)

    # — instantaneous CPU% for the process + its children —
    cpu_proc=$(
      {
        # main process
        ps -p "$PID" -o %cpu=;
        # any direct children
        for C in $(pgrep -P "$PID"); do
          ps -p "$C" -o %cpu=
        done
      } | awk '{s+=$1} END{printf "%.1f", s}'
    )

    # — RAM usage (RSS) in MB + percent —
    rss_kb=$(ps -p "$PID" -o rss= | tr -d ' ')
    ram_mb=$(awk "BEGIN{printf \"%.1f\", $rss_kb/1024}")
    ram_pct=$(awk "BEGIN{printf \"%.1f\", $rss_kb/$TOTAL_MEM_KB*100}")

    # — GPU utilization —
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i 0 | tr -d ' ')

    # — total VRAM used + percent —
    vram_used=$(nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits -i 0 | tr -d ' ')
    vram_pct=$(awk "BEGIN{printf \"%.1f\", $vram_used/$TOTAL_VRAM_MB*100}")

    # — per-core CPU usage —
    readarray -t core_vals < <(python - <<'PYCODE'
import psutil
for pct in psutil.cpu_percent(percpu=True, interval=0.1):
    print(f"{pct:.1f}")
PYCODE
    )
    core_str=""
    for idx in "${!core_vals[@]}"; do
      core_str+=" ${idx}:${core_vals[idx]}%"
    done

    # Append to logfile
    printf "%-19s | %6s | %8s | %6s | %6s | %8s | %6s |%s\n" \
      "$TS" "$cpu_proc" "$ram_mb" "$ram_pct" "$gpu_util" \
      "$vram_used" "$vram_pct" "$core_str" \
      >> "$LOGFILE"

    sleep "$INTERVAL"
  done

  echo "Process $PID finished; metrics saved to $LOGFILE"
}
