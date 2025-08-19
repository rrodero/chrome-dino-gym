"""Integration tests for the complete system."""

import numpy as np

from chrome_dino_gym import ChromeDinoEnv
from chrome_dino_gym.utils import benchmark_env, create_env, get_env_info


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

    def test_complete_episode(self):
        """Test a complete episode from start to finish."""
        env = create_env("ChromeDino-v0", max_episode_steps=100)

        obs, info = env.reset(seed=42)
        initial_score = info["score"]

        episode_reward = 0
        episode_length = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Verify observation shape consistency
            assert obs.shape == (20,)

            # Verify info dict has expected keys
            assert "score" in info
            assert "obstacles_passed" in info
            assert "step_count" in info

            if terminated or truncated:
                break

        # Episode should have progressed
        assert episode_length > 0
        assert info["score"] >= initial_score

        env.close()

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = create_env("ChromeDino-v0", max_episode_steps=50)

        episode_scores = []

        for _episode in range(3):
            obs, info = env.reset()

            while True:
                action = np.random.choice([0, 1, 2])
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    episode_scores.append(info["score"])
                    break

        # Should have completed all episodes
        assert len(episode_scores) == 3
        # All scores should be non-negative
        assert all(score >= 0 for score in episode_scores)

    def test_deterministic_behavior(self):
        """Test that environment behaves deterministically with same seed."""

        def run_episode(seed):
            env = create_env("ChromeDino-v0", max_episode_steps=20)
            obs, info = env.reset(seed=seed)

            observations = [obs.copy()]
            actions = [0, 1, 0, 2, 0]  # Fixed action sequence

            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                observations.append(obs.copy())

                if terminated or truncated:
                    break

            env.close()
            return observations

        # Run same sequence twice with same seed
        obs1 = run_episode(42)
        obs2 = run_episode(42)

        # Should be identical
        assert len(obs1) == len(obs2)
        for o1, o2 in zip(obs1, obs2, strict=False):
            np.testing.assert_array_almost_equal(o1, o2)

    def test_benchmark_function(self):
        """Test the benchmark utility function."""
        results = benchmark_env("ChromeDino-v0", episodes=3)

        assert isinstance(results, dict)
        assert "episodes" in results
        assert "avg_length" in results
        assert "avg_reward" in results
        assert "avg_score" in results

        assert results["episodes"] == 3
        assert results["avg_length"] > 0
        assert results["avg_score"] >= 0

    def test_env_info_function(self):
        """Test the get_env_info utility function."""
        info = get_env_info(create_env("ChromeDino-v0"))

        assert isinstance(info, dict)
        assert "action_space" in info
        assert "observation_space" in info
        assert "metadata" in info
