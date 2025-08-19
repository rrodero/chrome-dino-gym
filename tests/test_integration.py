"""Integration tests for the complete system."""

import numpy as np

from chrome_dino_gym import ChromeDinoEnv
from chrome_dino_gym.utils import create_env


class TestIntegration:
    """Integration tests for the complete Chrome Dino Gym system."""

    def test_make_function(self):
        """Test environment creation via make function."""
        env = create_env("ChromeDino-v0")

        assert isinstance(env.unwrapped, ChromeDinoEnv)

        # Test basic functionality
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)

        env.close()
