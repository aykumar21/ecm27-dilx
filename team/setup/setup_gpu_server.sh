#!/usr/bin/env bash
# ECM 2027 Sim Track — GPU server bootstrap (Ubuntu 22.04/24.04, x86_64, NVIDIA GPU).
# Idempotent: safe to re-run. Each stage can be skipped with SKIP_<STAGE>=1.
#
#   bash setup_gpu_server.sh                 # full setup into ~/ecm27
#   WORKDIR=/data/ecm27 bash setup_gpu_server.sh
#   SKIP_APT=1 bash setup_gpu_server.sh      # no sudo available / already installed
#
# Stages: preflight -> apt -> clone+LFS -> python3.12 venv -> Isaac Sim/Lab install
#         -> local eval config -> checks -> (optional) dataset sample download
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/ecm27}"
# Team private repo (needs auth on the server: `gh auth login` or a fine-grained PAT).
REPO_URL="${REPO_URL:-https://github.com/aykumar21/ecm27-dilx.git}"
UPSTREAM_URL="https://github.com/rocochallenge/ecm2027_sim_track.git"
REPO_DIR="$WORKDIR/ecm27-dilx"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_DATASET="rocochallenge2025/ecm2027_sim_data"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
log() { printf '\n\033[1;36m[ecm27] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[ecm27][warn] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[ecm27][fail] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. preflight
log "Preflight"
[[ "$(uname -s)/$(uname -m)" == "Linux/x86_64" ]] || die "Linux x86_64 required"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRV_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
# install_env.sh pulls torch cu130 wheels -> CUDA 13.0 runtime needs driver branch >= 580.
(( DRV_MAJOR >= 580 )) || warn "Driver $DRV_MAJOR < 580: torch cu130 / Isaac Sim 6 will likely fail. Upgrade the driver."
. /etc/os-release && echo "OS: $PRETTY_NAME"
mkdir -p "$WORKDIR"
FREE_GB=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc 0-9)
echo "Free disk at $WORKDIR: ${FREE_GB} GB"
(( FREE_GB >= 80 )) || warn "<80 GB free. Isaac Sim + deps need ~40-60 GB; each dataset episode is ~3 GB (full dataset ~1.5 TB)."

# ---------------------------------------------------------------- 1. system packages
if [[ "${SKIP_APT:-0}" != 1 ]]; then
  log "Installing system packages (sudo)"
  sudo apt-get update
  sudo apt-get install -y git git-lfs build-essential cmake pkg-config curl \
    libgl1 libegl1 libglx0 libglib2.0-0 libsm6 libice6 libxrender1 \
    libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxkbcommon0 \
    libvulkan1 vulkan-tools ffmpeg xvfb
fi
git lfs install

# ---------------------------------------------------------------- 2. repo + LFS assets
log "Cloning team repo"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only || warn "pull failed (local changes?) — continuing with current checkout"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" remote get-url upstream >/dev/null 2>&1 || git -C "$REPO_DIR" remote add upstream "$UPSTREAM_URL"
git -C "$REPO_DIR" lfs pull

# ---------------------------------------------------------------- 3. Python 3.12 venv
log "Creating Python 3.12 venv at $REPO_DIR/.venv"
cd "$REPO_DIR"
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null && python3.12 -c 'import venv, ensurepip' 2>/dev/null; then
    python3.12 -m venv .venv
  else
    # Ubuntu 22.04 has no python3.12 package; use uv's managed CPython instead of deadsnakes.
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    uv python install 3.12
    uv venv --python 3.12 --seed .venv
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version'

# ---------------------------------------------------------------- 4. Isaac Sim 6.0.1 + Isaac Lab + project
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  log "Running official installer (downloads Isaac Sim, torch cu130, Isaac Lab — 20-40 min)"
  bash scripts/install_env.sh
  python -m pip install -r requirements/policy-grpc.txt
  python -m pip install pytest "huggingface_hub[cli]"
fi

# ---------------------------------------------------------------- 5. local eval config
log "Writing local eval config (fixes hard-coded /home/pine/... scene path)"
mkdir -p configs/evaluation/local
python - "$REPO_DIR" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
cfg = json.loads((root / "configs/evaluation/chamber_server.example.json").read_text())
cfg["layout"] = str(root / "robofinals/data/assets/Scene/scene04_1.usd")
out = root / "configs/evaluation/local/chamber_server.local.json"
out.write_text(json.dumps(cfg, indent=2) + "\n")
print(out); print(json.dumps(cfg, indent=2))
PY

# ---------------------------------------------------------------- 6. checks
log "Environment preflight (scripts/check_environment.py)"
python scripts/check_environment.py || warn "check_environment reported FAIL lines above — fix before running the simulator"
log "gRPC protocol unit test (no Isaac needed)"
python -m pytest -q robofinals/tests/test_grpc_policy.py || warn "gRPC test failed"

# ---------------------------------------------------------------- 7. dataset sample
if [[ "${SKIP_DATA:-0}" != 1 ]]; then
  log "Downloading dataset metadata + 1 sample LeRobot episode to $DATA_DIR (full set is ~1.5 TB — NOT downloaded)"
  mkdir -p "$DATA_DIR"
  hf download "$HF_DATASET" --repo-type dataset --local-dir "$DATA_DIR/ecm2027_sim_data" \
     --include "meta/*" "data/*" "videos/*" \
   || huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DATA_DIR/ecm2027_sim_data" \
     --include "meta/*" "data/*" "videos/*" \
   || warn "dataset download failed (run 'hf auth login' if the repo is gated)"
fi

cat <<EOF

=====================================================================
 Setup finished.  Repo: $REPO_DIR   venv: $REPO_DIR/.venv
 Next (see SETUP.md):
   cd $REPO_DIR && source .venv/bin/activate
   export OMNI_KIT_ACCEPT_EULA=YES     # only after you've read & accepted the NVIDIA Omniverse EULA
   bash team/setup/smoke_test_sim.sh   # boots Isaac Sim headless + example policy
=====================================================================
EOF
