"""Chrome Dino Gymnasium Environment Package."""

__version__ = "0.1.0"
__author__ = "Rafael Rodero"
__email__ = "rrodero@live.com"

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from .envs.dino_env import ChromeDinoEnv

register(
    id="ChromeDino-v0",
    entry_point="chrome_dino_gym.envs:ChromeDinoEnv",
    max_episode_steps=10000,
    reward_threshold=1000.0,
)


def make(env_id: str, **kwargs: Any) -> gym.Env[np.ndarray, int]:
    """Create environment instance."""
    import gymnasium as gym

    return gym.make(env_id, **kwargs)


__all__ = ["ChromeDinoEnv", "make"]
