# Team DIL-X: ECM 2027 Sim Track

Everything outside `team/` is the upstream starter code (`upstream` remote, rocochallenge/ecm2027_sim_track).
Keep our changes inside `team/` so upstream merges stay conflict-free.

**Plan and task split: [`team/PLAN.md`](PLAN.md).**

| Path | Purpose |
|---|---|
| `team/setup/` | GPU server bootstrap (`setup_gpu_server.sh`), simulator smoke test, `SETUP.md` (task, data, interface notes) |
| `team/policy/` | `policy.py` (`TeamPolicy`: model goes here), `client.py` (gRPC submission entrypoint) |
| `team/training/` | Data loading (HDF5 demos / LeRobot), training scripts |
| `team/tools/` | `mock_eval_server.py`: real gRPC protocol, fake env, no Isaac Sim needed |
| `team/docker/` | Submission `Dockerfile` + allow-list `.dockerignore` |

```bash
# Pull organiser updates
git fetch upstream && git merge upstream/main

# Local policy test without the simulator
PYTHONPATH=. python team/tools/mock_eval_server.py --num-envs 2 &
ECM27_EVAL_ENDPOINT=127.0.0.1:50000 PYTHONPATH=. python -m team.policy.client

# Build submission image
docker build -f team/docker/Dockerfile -t ecm27-policy:latest .
```

Not in git: datasets (`data/`), checkpoints, eval results, `.venv/`, `.deps/`.
