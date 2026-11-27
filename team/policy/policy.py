"""Team DIL-X policy. Replace the body of `act` with model inference.

Contract (README "Participant policy interface"):
  * act() receives N decoded observations and returns float32 [N, 19] (or [N, 20] legacy; never mix).
  * reset(i) is called when env i reports terminated/truncated: clear any history for that env.
"""
from __future__ import annotations

import numpy as np

ACTION_DIM = 19  # [0:7] L pose xyz+qxyzw, [7:14] R pose, [14:16] grippers (<0 close), [16:19] base vx,vy,wz


def legacy_wheels_to_base(w: np.ndarray) -> np.ndarray:
    """Dataset 20-D action -> 19-D base command (README formula). w: [..., 4] -> [..., 3].
    Note: 20-D inputs bypass the simulator's alpha=0.1 base smoothing; 19-D inputs do not."""
    w1, w2, w3, w4 = np.moveaxis(w, -1, 0)
    return np.stack([(w1 + w2 + w3 + w4) / 4, (-w1 + w2 + w3 - w4) / 4, (-w1 + w2 - w3 + w4) / 4], -1)


class TeamPolicy:
    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self._hold: dict[int, np.ndarray] = {}  # per-env state example

    def reset(self, env_index: int) -> None:
        self._hold.pop(env_index, None)

    def act(self, observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        """Placeholder: hold the current EEF poses, grippers open, base still."""
        actions = np.zeros((len(observations), ACTION_DIM), dtype=np.float32)
        for i, obs in enumerate(observations):
            if i not in self._hold:
                # TODO: eef_pos is world-axis relative to env origin, actions are in the robot ROOT frame;
                # convert with the base pose before using this as a hold target on a moving base.
                self._hold[i] = np.concatenate([obs["eef_pos"][0], obs["eef_quat"][0],
                                                obs["eef_pos"][1], obs["eef_quat"][1]]).astype(np.float32)
            actions[i, :14] = self._hold[i]
            if not np.any(actions[i, 3:7]):
                actions[i, 6] = 1.0
            if not np.any(actions[i, 10:14]):
                actions[i, 13] = 1.0
        return actions
