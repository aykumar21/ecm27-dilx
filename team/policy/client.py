"""Submission entrypoint: gRPC client loop (same protocol handling as examples/policy_docker/example_policy.py)."""
import os
import sys
import time

import grpc
import numpy as np

from robofinals.distributed import policy_v1_pb2 as pb
from robofinals.distributed.grpc_policy import PolicyClient
from team.policy.policy import TeamPolicy

FIELD_SHAPES = {
    "first_person_camera_rgb": (np.uint8, (480, 640, 3)),
    "left_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "right_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "actions": (np.dtype("<f4"), (19,)),
    "joint_pos": (np.dtype("<f4"), (77,)),
    "joint_vel": (np.dtype("<f4"), (77,)),
    "eef_pos": (np.dtype("<f4"), (2, 3)),
    "eef_quat": (np.dtype("<f4"), (2, 4)),
    "gripper_pos": (np.dtype("<f4"), (2,)),
}
TIMEOUT = 300 * 60


def decode(observation):
    return {name: np.frombuffer(getattr(observation, name), dtype=dtype).reshape(shape).copy()
            for name, (dtype, shape) in FIELD_SHAPES.items()}


def main():
    endpoint = os.environ["ECM27_EVAL_ENDPOINT"]
    channel = grpc.insecure_channel(endpoint, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
    grpc.channel_ready_future(channel).result(timeout=TIMEOUT)
    client = PolicyClient(channel)
    policy = TeamPolicy(checkpoint=os.environ.get("ECM27_CHECKPOINT", "/app/checkpoints"))
    transition = client.get_observation(pb.Empty(), timeout=TIMEOUT)
    steps, t0 = 0, time.time()
    while True:
        for i, (terminated, truncated) in enumerate(zip(transition.terminated, transition.truncated)):
            if terminated or truncated:
                policy.reset(i)
        observations = [decode(obs) for obs in transition.observations]
        actions = np.asarray(policy.act(observations), dtype=np.float32)
        if not np.all(np.isfinite(actions)):
            actions = np.nan_to_num(actions)  # non-finite actions are rejected with INVALID_ARGUMENT
        request = pb.ActionBatch(actions=[pb.Action(values=row) for row in actions])
        try:
            transition = client.step(request, timeout=TIMEOUT)
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise
            transition = client.get_observation(pb.Empty(), timeout=TIMEOUT)
        steps += 1
        if steps % 500 == 0:
            print(f"[team-policy] {steps} steps, {steps / (time.time() - t0):.1f} steps/s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
