"""Tests for ChromeDinoEnv"""

import gymnasium as gym
import numpy as np
import pytest

from chrome_dino_gym import ChromeDinoEnv


class TestChromeDinoEnv:
    """Test cases for ChromeDinoEnv"""

    def test_initialization(self):
        """Test environment initialization"""
        env = ChromeDinoEnv()

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (20,)

    def test_reset(self, env, random_seed):
        """Test environment reset"""
        obs, info = env.reset(seed=random_seed)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (20,)
        assert isinstance(info, dict)
        assert "score" in info
        assert info["score"] == 0

    def test_step(self, env):
        """Test environment step function"""
        env.reset()

        # Test valid actions
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert obs.shape == (20,)
            assert isinstance(reward, int | float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            if terminated or truncated:
                break

    def test_invalid_action(self, env):
        """Test invalid action handling"""
        env.reset()

        with pytest.raises(AssertionError):
            env.step(3)

        with pytest.raises(AssertionError):
            env.step(-1)

    def test_episode_termination(self, env):
        """Test episode termination conditions"""
        env.reset()

        # Run until termination
        steps = 0
        max_steps = 1000

        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated:
                # Should terminate due to collision
                assert info.get("game_over_reason") == "collision"
                break
            elif truncated:
                # Should terminate due to max steps
                break

        assert steps <= max_steps

    def test_reward_calculation(self, env):
        """Test reward calculation"""
        env.reset()

        # Take a few steps and check rewards
        total_reward = 0
        for _ in range(10):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Should habe received some positive rewards for surviving
        assert total_reward > 0

    def test_observation_bounds(self, env):
        """Test observation values are within expected bounds"""
        env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Most values should be within reasonable bounds
            assert np.all(obs >= -5.0), f"Observation values too low: {obs}"
            assert np.all(obs <= 5.0), f"Observation values too high: {obs}"

            if terminated or truncated:
                break

    def test_render_modes(self):
        """Test different render modes"""
        # Test rgb_array mode
        env_rgb = ChromeDinoEnv(render_mode="rgb_array")
        env_rgb.reset()

        # Should return rgb array
        rgb_array = env_rgb.render()
        assert isinstance(rgb_array, np.ndarray)
        assert len(rgb_array.shape) == 3

        env_rgb.close()

        # Test human mode (should not return anything)
        env_human = ChromeDinoEnv(render_mode="human")
        env_human.reset()

        result = env_human.render()
        assert result is None

        env_human.close()

    def test_custom_reward_config(self):
        """Test custom reward configuration"""
        custom_rewards = {
            "step_reward": 0.5,
            "obstacle_reward": 20.0,
            "collision_penalty": -50.0,
        }

        env = ChromeDinoEnv(reward_config=custom_rewards)
        env.reset()

        # Check that custom rewards are used
        assert env.reward_config["step_reward"] == 0.5
        assert env.reward_config["obstacle_reward"] == 20.0
        assert env.reward_config["collision_penalty"] == -50.0

        env.close()
