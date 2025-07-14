#!/usr/bin/env bash
# logger.sh ---------------------------------------------------------
# Logs CPU, RAM, and GPU metrics (NVIDIA or AMD) for a given PID.
# Usage:
#   source logger.sh
#   start_metrics_logger <PID> [INTERVAL] <LOGFILE>

start_metrics_logger() {
  local PID="$1"
  local INTERVAL="${2:-60}"
  local LOGFILE="$3"

  ########################
  # Detect GPU back-end
  ########################
  local GPU_BACKEND="none" # nvidia/amd/none
  local TOTAL_VRAM_MB=""

  if command -v nvidia-smi &>/dev/null; then
    GPU_BACKEND="nvidia"
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total \
                    --format=csv,noheader,nounits -i 0 | tr -d ' ')
  elif command -v rocm-smi &>/dev/null; then
    GPU_BACKEND="amd"
    # Try rocm-smi first
    TOTAL_VRAM_MB=$(rocm-smi --showmemuse -d 0          \
                     | awk '/Total VRAM/ {print $(NF-1); exit}')
    # fall back to sysfs if parsing failed
    if [[ -z "$TOTAL_VRAM_MB" && -r /sys/class/drm/card0/device/mem_info_vram_total ]]; then
      TOTAL_VRAM_MB=$(( $(cat /sys/class/drm/card0/device/mem_info_vram_total) / 1024 / 1024 ))
    fi
  fi

  ########################
  # System RAM total
  ########################
  local TOTAL_MEM_KB
  TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

  ########################
  # Log header
  ########################
  {
    printf "%-19s | %6s | %8s | %6s | %6s | %8s | %6s | %s\n" \
      "Timestamp" "CPU%" "RAM(MB)" "RAM%" "GPU0%" "VRAM(MB)" "VRAM%" "Per-Core(%)"
    printf '%.0s─' {1..120}; echo
  } >> "$LOGFILE"

  ########################
  # Main loop
  ########################
  while kill -0 "$PID" 2>/dev/null; do
    local TS=$(date +%Y-%m-%dT%H:%M:%S)

    # CPU (proc + children)
    local cpu_proc=$(
      { ps -p "$PID" -o %cpu=;
        for C in $(pgrep -P "$PID"); do ps -p "$C" -o %cpu=; done; } \
      | awk '{s+=$1} END{printf "%.1f", s}'
    )

    # RAM
    local rss_kb=$(ps -p "$PID" -o rss= | tr -d ' ')
    local ram_mb=$(awk "BEGIN{printf \"%.1f\", $rss_kb/1024}")
    local ram_pct=$(awk "BEGIN{printf \"%.1f\", $rss_kb/$TOTAL_MEM_KB*100}")

    # GPU
    local gpu_util="-"  vram_used="-"  vram_pct="-"

    case "$GPU_BACKEND" in
      nvidia)
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                    --format=csv,noheader,nounits -i 0 | tr -d ' ')
        vram_used=$(nvidia-smi --query-gpu=memory.used \
                    --format=csv,noheader,nounits -i 0 | tr -d ' ')
        ;;
      amd)
        # Utilisation
        gpu_util=$(rocm-smi --showuse -d 0 \
                    | awk '/GPU use/ {gsub(/%/,""); print $(NF)}' | head -n1)
        # VRAM used (try rocm-smi first)
        vram_used=$(rocm-smi --showmemuse -d 0 \
                    | awk '/Used VRAM|VRAM Used|VRAM use/ {print $(NF-1); exit}')
        # Fallback to sysfs counter
        if [[ -z "$vram_used" && -r /sys/class/drm/card0/device/mem_info_vram_used ]]; then
          vram_used=$(( $(cat /sys/class/drm/card0/device/mem_info_vram_used) / 1024 / 1024 ))
        fi
        ;;
    esac

    # VRAM %
    if [[ -n "$TOTAL_VRAM_MB" && -n "$vram_used" && "$TOTAL_VRAM_MB" != "0" && "$vram_used" != "-" ]]; then
      vram_pct=$(awk "BEGIN{printf \"%.1f\", $vram_used/$TOTAL_VRAM_MB*100}")
    fi

    # Per-core CPU
    readarray -t core_vals < <(python - <<'PY'
import psutil, time
print('\n'.join(f"{p:.1f}" for p in psutil.cpu_percent(percpu=True, interval=0.1)))
PY
    )
    local core_str=""
    for idx in "${!core_vals[@]}"; do core_str+=" ${idx}:${core_vals[idx]}%"; done

    # Write line
    printf "%-19s | %6s | %8s | %6s | %6s | %8s | %6s |%s\n" \
      "$TS" "$cpu_proc" "$ram_mb" "$ram_pct" "$gpu_util" \
      "$vram_used" "$vram_pct" "$core_str" >> "$LOGFILE"

    sleep "$INTERVAL"
  done

  echo "Process $PID finished; metrics saved to $LOGFILE"
}
