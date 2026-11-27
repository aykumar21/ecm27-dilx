# ECM 2027 Sim Track: GPU server setup

Starter repo: `rocochallenge/ecm2027_sim_track` (commit `17159e5`, 23 Nov 2026).
Deadline: **30 Dec 2026, 12:00 SGT**.

## 1. What the task is

| Item | Detail |
|---|---|
| Robot | Galbot-Mobile (G1): wheeled base, 2 × 7-DoF arms, 2 grippers, head camera |
| Simulator | Isaac Sim 6.0.1 + Isaac Lab (rev `28a37ce`), PhysX, scene `scene04_1.usd` |
| Task | `ChamberServiceTask` (chamber preventive maintenance) |
| Scoring (13 pts max) | open chamber 2, extract old part 1, place it in the box 1, sponge-wipe 5 faces 6, place replacement 2, close chamber 1. Stages unlock in order; each point counts once per episode |
| Episode limits | 900 s sim time (`truncated`), 300 min wall clock, control dt 0.02 s (50 Hz) |
| What you submit | A Linux amd64 **Docker image** running a gRPC *client*, with CPU/GPU/VRAM requirements stated. NTU runs the simulator and sends feedback within 5 working days, and you can resubmit |

**Observations, per env:** head RGB `480×640×3`, left and right wrist RGB `224×224×3`, `joint_pos`/`joint_vel` `[77]` (relative to defaults), `eef_pos` `[2,3]`, `eef_quat` `[2,4]` xyzw, `gripper_pos` `[2]`, last `actions` `[19]`.

**Action (19-D, preferred):** `[0:7]` left EEF pose and `[7:14]` right EEF pose (`xyz + qxqyqzqw`, robot root frame, absolute); `[14],[15]` grippers (<0 closes); `[16:19]` base `vx, vy, wz` (EMA-smoothed with α = 0.1). The dataset's 20-D legacy actions use 4 wheel values for the base. **Don't truncate 20-D to 19-D.** Either send 20-D, or convert with `vx=(w1+w2+w3+w4)/4`, `vy=(-w1+w2+w3-w4)/4`, `wz=(-w1+w2-w3+w4)/4`. Be careful with the conversion, because legacy actions skip the smoothing.

The policy has to handle a batch of N envs (N is set by the evaluator, 1–27) and reset per-env state when an env reports `terminated` or `truncated`.

## 2. Dataset (`rocochallenge2025/ecm2027_sim_data`, about 1.55 TB)

- `meta/`, `data/`, `videos/`: a **LeRobot v3.0** sample, 1 episode, 12,572 frames at 30 fps. It contains state[77], action[20], and 3 H.264 cameras.
- `Roco/batch1/` (173 demos), `Roco/batch2/`, and batch 3 on 28 Nov: each `ChamberServiceTask_<id>/galbot_success.hdf5` is **about 2.9 GB**, plus `running_args.json`.
- Batch 1 alone is about 500 GB, so put the data on a large disk (`DATA_DIR=/big/disk`). Download selectively:
  ```bash
  hf download rocochallenge2025/ecm2027_sim_data --repo-type dataset \
     --local-dir $DATA_DIR/ecm2027_sim_data --include "Roco/batch1/ChamberServiceTask_17931*/*"
  ```
- Replay a demo in sim to check it: `./scripts/run_replay_eval.sh /abs/path/galbot_success.hdf5 action`

## 3. Server requirements

Ubuntu 22.04/24.04 x86_64. NVIDIA RTX-class GPU (Isaac Sim needs RT cores; A100/H100 can't render cameras). **Driver ≥ 580**, because the installer uses torch `cu130`. Plan for ≥ 32 GB RAM and ≥ 100 GB free disk for software, plus space for data. sudo is needed for apt. Docker and nvidia-container-toolkit are needed to build and test the submission image.

## 4. Steps

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/aykumar21/ecm27-dilx.git ~/ecm27/ecm27-dilx   # needs gh auth / PAT
bash ~/ecm27/ecm27-dilx/team/setup/setup_gpu_server.sh                  # ~30-60 min, mostly Isaac Sim download
#   options: WORKDIR=/data/ecm27  DATA_DIR=/data/ecm27/data  SKIP_APT=1  SKIP_DATA=1
cd ~/ecm27/ecm27-dilx && source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES             # after reading NVIDIA's Omniverse EULA
bash team/setup/smoke_test_sim.sh        # real simulator + example policy
```

`setup_gpu_server.sh` does the following:
- checks the driver and disk
- installs apt packages
- clones the repo and pulls LFS assets (about 73 MB)
- creates a Python 3.12 venv (falls back to `uv` on 22.04)
- runs the official `scripts/install_env.sh`
- writes `configs/evaluation/local/chamber_server.local.json`, because the example config has a hard-coded `/home/pine/...` scene path
- runs `check_environment.py` and the gRPC unit test
- downloads only the LeRobot sample episode

**README discrepancy:** the README says to clone `LightwheelAI/RoCo-SG27-x-Lightwheel`, but `install_env.sh` installs whatever checkout it lives in. The script uses the repo from the organisers' email. If something looks missing, ask on Discord.

## 5. Developing without the simulator

`mock_eval_server.py` serves the real gRPC protocol with random, correctly shaped observations. Use it to test your client and Docker image on any machine:

```bash
cd ecm27-dilx
PYTHONPATH=. python team/tools/mock_eval_server.py --num-envs 2 --episode-steps 200 --host 0.0.0.0
ECM27_EVAL_ENDPOINT=127.0.0.1:50000 PYTHONPATH=. python examples/policy_docker/example_policy.py
```

Testing the policy image against the real simulator, as the evaluator runs it:
```bash
docker network create ecm27 || true
# simulator on host: --ipc_host=0.0.0.0 ; container reaches it via host gateway
docker build -f team/docker/Dockerfile -t ecm27-policy:latest .
docker run --rm --gpus all --add-host=host.docker.internal:host-gateway \
  -e ECM27_EVAL_ENDPOINT=host.docker.internal:50000 ecm27-policy:latest
docker save -o ecm27-policy.tar ecm27-policy:latest      # deliverable
```
When you add model files, also add them to `Dockerfile.dockerignore`, which is an allowlist.

## 6. Troubleshooting
- `check_environment` FAIL on Scene or `galbot_*` with "LFS pointers": run `git lfs pull` again.
- Simulator exits with no display: run it under `xvfb-run -a`. The env server forces a non-headless Kit UI.
- `INVALID_ARGUMENT` on Step: the action has the wrong dimension, contains NaN or Inf, has a quaternion norm below 1e-6, or a position beyond ±10 m.
- `FAILED_PRECONDITION`: a wall-clock reset happened. Call `GetObservation` again (the example client already does this).
