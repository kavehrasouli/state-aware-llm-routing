#!/usr/bin/env bash
# Cost-model sweep: 11 cells (9 backbone + 2 asymmetric) x 3 repetitions on one node.
# R is measured per run as prompt + generated tokens (R_actual); fit on R_actual, not R_target.
# With --ignore-eos every request generates its full max_tokens.
# Prerequisite: python3 generate_grid_cells.py. Run from benchmarks/.
set -euo pipefail

PI_HOST="admin@192.168.2.5"
BASE_URL="http://192.168.2.5:8080"
MODEL="qwen2.5-0.5b"
TEMP_GATE_C=60
THERMAL_LOG="cost_model_sweep_thermal.log"   # lives in $PI_HOST's home dir
MAX_RETRIES=5
REPS=3
MANIFEST_TSV="grid_cells/manifest.tsv"

# All outputs (CSV, raw JSONs, grid cells used) go into one archive folder.
ARCHIVE_DIR="cost_model_results_pi-node-3_4B-8g"
RAW_RESULTS_DIR="$ARCHIVE_DIR/raw_results"

if [ ! -f "$MANIFEST_TSV" ]; then
  echo "FATAL: $MANIFEST_TSV not found -- run 'python3 generate_grid_cells.py' first."
  exit 1
fi

mkdir -p "$RAW_RESULTS_DIR"
cp -r grid_cells "$ARCHIVE_DIR/grid_cells"

# SSH password is read from a local, gitignored file (not stored in this script).
SSH_PASS_FILE=".pi_node3_ssh_pass"
if [ ! -f "$SSH_PASS_FILE" ]; then
  echo "FATAL: $SSH_PASS_FILE not found -- expected in benchmarks/, 600 permissions, password only."
  exit 1
fi
ssh_pi() {
  sshpass -f "$SSH_PASS_FILE" ssh -o StrictHostKeyChecking=accept-new "$PI_HOST" "$@"
}

CSV="$ARCHIVE_DIR/sweep_cost_model_summary.csv"
echo "block,cell_label,rep,B,R_target,R_actual,tpot_mean,ttft_mean,note,temp_peak,throttled,run_timestamp,json_file" > "$CSV"

echo "Starting thermal logger on pi-node-2 (2s interval, current-state throttle bits)..."
LOGGER_PID=$(ssh_pi "nohup bash -c '
  while true; do
    t=\$(vcgencmd measure_temp | grep -oE \"[0-9.]+\")
    th=\$(vcgencmd get_throttled | cut -d= -f2)
    echo \"\$(date +%s) \$t \$th\"
    sleep 2
  done
' > ~/$THERMAL_LOG 2>&1 < /dev/null & echo \$!")
echo "  remote logger PID: $LOGGER_PID"
sleep 3   # let the first samples land before we rely on them

cleanup() {
  echo "Stopping remote thermal logger (PID $LOGGER_PID)..."
  ssh_pi "kill $LOGGER_PID 2>/dev/null || true"
}
trap cleanup EXIT

wait_for_cool() {
  while true; do
    last_line=$(ssh_pi "tail -1 ~/$THERMAL_LOG")
    temp=$(echo "$last_line" | awk '{print $2}')
    if (( $(echo "$temp < $TEMP_GATE_C" | bc -l) )); then
      echo "  pi-node-2 at ${temp}°C — OK to start."
      break
    fi
    echo "  pi-node-2 at ${temp}°C — waiting to cool below ${TEMP_GATE_C}°C..."
    sleep 10
  done
}

# Prints "<peak_temp> <throttled 0|1>" for the [start_epoch, end_epoch] window.
# Same four current-state bits as sweep_router_signal.sh (0x1 under-voltage,
# 0x2 freq-capped, 0x4 throttled, 0x8 soft-temp-limit), same malformed-line
# guard before the arithmetic.
check_run_thermal() {
  local start_epoch=$1 end_epoch=$2
  ssh_pi "awk -v s=$start_epoch -v e=$end_epoch '\$1>=s && \$1<=e' ~/$THERMAL_LOG" > /tmp/sweep_window.log
  local peak_temp throttled_any
  peak_temp=$(awk '{print $2}' /tmp/sweep_window.log | sort -n | tail -1)
  throttled_any=0
  while read -r _ _ hex; do
    if [[ "$hex" =~ ^0x[0-9a-fA-F]+$ ]]; then
      val=$((hex))
      if (( val & 0xF )); then throttled_any=1; fi
    fi
  done < /tmp/sweep_window.log
  echo "$peak_temp $throttled_any"
}

n_cells=$(wc -l < "$MANIFEST_TSV" | tr -d ' ')
echo "Loaded $n_cells cells from $MANIFEST_TSV — $((n_cells * REPS)) total runs planned."

# Read the manifest on fd 3 so ssh inside the loop cannot consume it from stdin.
while IFS=$'\t' read -r block label B r_target cell_path note <&3; do
  for ((rep=1; rep<=REPS; rep++)); do
    echo ""
    echo "=== [$block] $label  B=$B R_target=$r_target  rep=$rep/$REPS ==="
    if [ -n "$note" ]; then
      echo "  NOTE: $note"
    fi

    attempt=0
    while true; do
      attempt=$((attempt+1))
      wait_for_cool

      # +%Y-%m-%dT%H:%M:%S works with both GNU and BSD (macOS) date; -Is is GNU-only.
      run_timestamp=$(date "+%Y-%m-%dT%H:%M:%S%z")
      start_epoch=$(date +%s)
      # `|| true` stops set -e from aborting, so the real exit code can be checked via PIPESTATUS.
      python3 bench.py --base-url "$BASE_URL" --model "$MODEL" \
        --dataset-name local --dataset-path "$cell_path" \
        --num-prompts "$B" --max-concurrency "$B" \
        --ignore-eos --save-result 2>&1 | tee /tmp/sweep_bench_output.txt || true
      bench_exit=${PIPESTATUS[0]}
      end_epoch=$(date +%s)

      if [ "$bench_exit" -ne 0 ]; then
        echo "  *** bench.py exited nonzero ($bench_exit) for $label rep=$rep -- a real crash,"
        echo "  *** not an expected partial-request-failure. Aborting sweep for investigation."
        exit 1
      fi

      json_file=$(grep "Saved → " /tmp/sweep_bench_output.txt | sed 's/^Saved → //')
      if [ -z "$json_file" ] || [ ! -f "$json_file" ]; then
        echo "  *** Could not find bench.py's 'Saved → <path>' line in its output --"
        echo "  *** aborting rather than guessing which file this run produced."
        exit 1
      fi

      # Move this run's result from bench.py's results/ folder into the archive.
      moved_json_file="$RAW_RESULTS_DIR/$(basename "$json_file")"
      mv "$json_file" "$moved_json_file"
      json_file="$moved_json_file"

      tpot_mean=$(python3 -c "import json; print(json.load(open('$json_file'))['tpot']['mean_ms'])")
      ttft_mean=$(python3 -c "import json; print(json.load(open('$json_file'))['ttft']['mean_ms'])")
      failed=$(python3 -c "import json; print(json.load(open('$json_file'))['failed'])")
      if [ "$failed" != "0" ]; then
        echo "  NOTE: $failed request(s) failed in this run (see $json_file) — not auto-retried, only throttle triggers a retry."
      fi

      # R_actual: sum of prompt + generated tokens over successful requests (the value to fit on).
      r_actual=$(python3 -c "
import json
d = json.load(open('$json_file'))
print(sum(r['actual_prompt_len'] + r['generated_output_tokens'] for r in d['raw'] if r['success']))
")

      read -r peak_temp throttled < <(check_run_thermal "$start_epoch" "$end_epoch")
      echo "  R_actual=${r_actual} (target ${r_target})  tpot_mean=${tpot_mean}ms  ttft_mean=${ttft_mean}ms  temp_peak=${peak_temp}C  throttled=${throttled}"

      if [ "$throttled" = "1" ]; then
        echo "  *** THROTTLED during this run (attempt $attempt/$MAX_RETRIES) — discarding, retrying with a fresh run after 30s cooldown."
        if [ "$attempt" -ge "$MAX_RETRIES" ]; then
          echo "  Giving up after $MAX_RETRIES attempts for $label rep=$rep — logging anyway, flagged throttled=1."
          break
        fi
        sleep 30
        continue
      fi
      break
    done

    echo "$block,$label,$rep,$B,$r_target,$r_actual,$tpot_mean,$ttft_mean,\"$note\",$peak_temp,$throttled,$run_timestamp,$json_file" >> "$CSV"
  done
done 3< "$MANIFEST_TSV"

echo ""
echo "Sweep complete. Summary written to $CSV"
echo "($((n_cells * REPS)) runs planned; check the CSV for any throttled=1 rows that hit MAX_RETRIES)"
