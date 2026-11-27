"""Mock ECM27 evaluator: serves the real gRPC protocol (ecm27.policy.v1.Policy) with a fake env.

Lets you develop and test a policy container/client WITHOUT Isaac Sim or a GPU.
Observations are random but have the exact official shapes/dtypes; episodes end after
--episode-steps steps (terminated) so per-env reset handling gets exercised.

Usage (from the ecm2027_sim_track repo root):
    PYTHONPATH=. python /path/to/mock_eval_server.py --port 50000 --num-envs 2
    ECM27_EVAL_ENDPOINT=127.0.0.1:50000 python examples/policy_docker/example_policy.py
"""
import argparse
from types import SimpleNamespace

import torch

from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS, JOINT_NAMES, OBSERVATION_SCHEMA
from robofinals.distributed.grpc_policy import GrpcPolicyServer


class MockEnv:
    def __init__(self, num_envs, episode_steps):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.t = torch.zeros(num_envs, dtype=torch.long)
        self.total_steps = 0
        self.action_space = SimpleNamespace(shape=(19,))
        self.unwrapped = SimpleNamespace(step_dt=0.02, scene={"robot": SimpleNamespace(data=SimpleNamespace(
            joint_names=JOINT_NAMES,
            default_joint_pos=torch.tensor(JOINT_DEFAULT_POS).repeat(num_envs, 1),
            default_joint_vel=torch.zeros((num_envs, 77))))})

    def _obs(self):
        policy = {}
        for name, dtype, shape in OBSERVATION_SCHEMA:
            if dtype == "uint8":
                policy[name] = torch.randint(0, 256, (self.num_envs, *shape), dtype=torch.uint8)
            else:
                policy[name] = torch.randn((self.num_envs, *shape), dtype=torch.float32) * 0.01
        return {"policy": policy}

    def reset(self, env_ids=None):
        if env_ids is None:
            self.t.zero_()
        else:
            self.t[env_ids] = 0
        return self._obs(), {}

    def step(self, action):
        assert action.shape == (self.num_envs, 19), action.shape
        self.total_steps += 1
        self.t += 1
        done = self.t >= self.episode_steps
        final = self._obs()
        self.t[done] = 0
        if self.total_steps % 100 == 0:
            print(f"[mock] {self.total_steps} steps; last action row0[:7]={action[0, :7].tolist()}", flush=True)
        return self._obs(), torch.zeros(self.num_envs), done, torch.zeros_like(done), {"final_obs": final}

    def close(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--episode-steps", type=int, default=200)
    a = p.parse_args()
    env = MockEnv(a.num_envs, a.episode_steps)
    print(f"[mock] serving ecm27.policy.v1.Policy on {a.host}:{a.port} with {a.num_envs} env(s)", flush=True)
    GrpcPolicyServer(lambda cfg: env, (a.host, a.port)).serve()


if __name__ == "__main__":
    main()
