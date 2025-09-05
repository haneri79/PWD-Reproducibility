#!/usr/bin/env bash
set -euo pipefail
CONF=${1:-hero/HS1_softCoulomb_3D/config.yaml}
LOGDIR=hero/HS1_softCoulomb_3D/logs
mkdir -p "$LOGDIR"
BASELINE_SOLVER=${BASELINE_SOLVER:-baseline_solver}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores
run() {
  local tag=$1
  local log="$LOGDIR/base_${tag}.log"
  echo "case_id=${tag}" > "$log"
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ" >> "$log"
  echo "cpu_model=$(lscpu | awk -F: '/Model name/{print $2}' | xargs)" >> "$log"
  echo "ram_gb=$(awk '/MemTotal/{printf \"%.1f\", $2/1024/1024}' /proc/meminfo)" >> "$log"
  ts_plan=$(date +%s.%N)
  $BASELINE_SOLVER --make-plans --config "$CONF" >/dev/null 2>&1 || true
  te_plan=$(date +%s.%N)
  awk -v a="$ts_plan" -v b="$te_plan" 'BEGIN{printf "offline_plan_s=%.6f\n", (b-a)}' >> "$log"
  ( while true; do
      ts=$(date +%s)
      e=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null || echo 0)
      echo "rapl_uj_ts=${ts} rapl_uj=${e}" >> "$LOGDIR/.pow_base_${tag}.tmp"
      sleep 1
    done ) & EM_PID=$!
  ts=$(date +%s.%N)
  /usr/bin/time -v $BASELINE_SOLVER --config "$CONF" 2> "$LOGDIR/.time_base_${tag}.tmp" | tee -a "$log"
  te=$(date +%s.%N)
  kill $EM_PID || true
  awk -v a="$ts" -v b="$te" 'BEGIN{printf "wall_time_s=%.6f\n", (b-a)}' >> "$log"
  awk '/Maximum resident set size/ {printf "peak_ram_gb=%.3f\n", $6/1024/1024}' "$LOGDIR/.time_base_${tag}.tmp" >> "$log"
  python - <<'PY' >>"$log"
import glob
uj=0.0
for f in glob.glob("hero/HS1_softCoulomb_3D/logs/.pow_base_*.tmp"):
    with open(f) as h:
        for ln in h:
            if "rapl_uj=" in ln:
                try: uj=float(ln.split("rapl_uj=")[1])
                except: pass
print(f"energy_wh={uj/3.6e9:.6f}")
PY
  echo "n_cores=${OMP_NUM_THREADS}" >> "$log"
  date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" >> "$log"
  echo "offline_included=full" >> "$log"
}
run "dtgrid"
