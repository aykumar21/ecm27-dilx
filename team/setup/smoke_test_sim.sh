#!/usr/bin/env bash
# Boots the real Chamber Service evaluator (Isaac Sim) and drives it with the example policy.
# Pass = server logs "listening", the client connects and steps without errors.
#   bash smoke_test_sim.sh [repo_dir]      (default ~/ecm27/ecm2027_sim_track)
#   STEPS_SECONDS=600 bash smoke_test_sim.sh   # let the client run longer
set -euo pipefail
REPO_DIR="${1:-$HOME/ecm27/ecm27-dilx}"
STEPS_SECONDS="${STEPS_SECONDS:-180}"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR:$REPO_DIR/vendor/isaaclab_arena${PYTHONPATH:+:$PYTHONPATH}"
[[ -n "${OMNI_KIT_ACCEPT_EULA:-}" ]] || { echo "Set OMNI_KIT_ACCEPT_EULA=YES after reading the NVIDIA Omniverse EULA."; exit 1; }
mkdir -p logs
CFG=configs/evaluation/local/chamber_server.local.json

# The env server forces a non-headless Kit UI, so a display is required: use Xvfb on a headless box.
RUN=( ./scripts/run_eval_env.sh --remote_protocol=grpc --eval_env_config "$CFG"
      --ipc_host=127.0.0.1 --ipc_port=50000 --eval_record_dir ./eval_result/smoke )
if [[ -z "${DISPLAY:-}" ]]; then RUN=( xvfb-run -a "${RUN[@]}" ); fi
echo "[smoke] starting simulator: ${RUN[*]}  (first boot compiles shaders: 5-15 min)"
"${RUN[@]}" > logs/eval_server.log 2>&1 &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null || true; pkill -P $SIM_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do   # wait up to 30 min for the gRPC port
  grep -q "listening on" logs/eval_server.log 2>/dev/null && break
  kill -0 $SIM_PID 2>/dev/null || { echo "[smoke] simulator exited early:"; tail -50 logs/eval_server.log; exit 1; }
  sleep 10
done
grep -q "listening on" logs/eval_server.log || { echo "[smoke] timed out waiting for server"; tail -50 logs/eval_server.log; exit 1; }

echo "[smoke] server up; running example policy for ${STEPS_SECONDS}s"
set +e
ECM27_EVAL_ENDPOINT=127.0.0.1:50000 timeout "$STEPS_SECONDS" python -u examples/policy_docker/example_policy.py > logs/example_policy.log 2>&1
RC=$?
set -e
if [[ $RC -eq 124 ]]; then echo "[smoke] PASS: policy loop ran ${STEPS_SECONDS}s without error"; else
  echo "[smoke] FAIL: policy exited with $RC"; tail -30 logs/example_policy.log; exit 1; fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
echo "[smoke] logs: $REPO_DIR/logs/, recordings: $REPO_DIR/eval_result/smoke/"
