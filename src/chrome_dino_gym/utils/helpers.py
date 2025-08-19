"""Helper functions for environment creation and management"""

from typing import Any

import gymnasium as gym

from ..envs import ChromeDinoEnv


def create_env(env_id: str = "ChromeDino-v0", **kwargs: Any) -> ChromeDinoEnv:
    """
    Create a Chrome Dino environment instance.

    Args:
        env_id: Environment ID (default: "ChromeDino-v0")
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Configured ChromDinoEnv instance
    """
    # Register the environment if not already registered
    try:
        env = gym.make(env_id, **kwargs)
    except gym.error.UnregisteredEnv:
        register_envs()
        env = gym.make(env_id, **kwargs)

    # Ensure we return the correct type
    if isinstance(env, ChromeDinoEnv):
        return env
    else:
        # This should not happen with proper registration, but for type safety
        raise TypeError(f"Expected ChromeDinoEnv, got {type(env)}")


def register_envs() -> None:
    """Register all Chrome Dino environments with Gymnasium"""
    from gymnasium.envs.registration import register

    # Register main environment
    register(
        id="ChromeDino-v0",
        entry_point="chrome_dino_gym.envs:ChromeDinoEnv",
        max_episode_steps=10000,
        reward_threshold=1000.0,
    )

    # Register variant with different settings
    register(
        id="ChromeDino-Easy-v0",
        entry_point="chrome_dino_gym.envs:ChromeDinoEnv",
        max_episode_steps=5000,
        reward_threshold=500.0,
        kwargs={
            "reward_config": {
                "step_reward": 0.2,
                "obstacle_reward": 15.0,
                "collision_penalty": -50.0,
            }
        },
    )

    register(
        id="ChromeDino-Hard-v0",
        entry_point="chrome_dino_gym.envs:ChromeDinoEnv",
        max_episode_steps=20000,
        reward_threshold=2000.0,
        kwargs={
            "reward_config": {
                "step_reward": 0.05,
                "obstacle_reward": 5.0,
                "collision_penalty": -200.0,
            }
        },
    )


def get_env_info(env: ChromeDinoEnv) -> dict[str, Any]:
    """
    Get information about an environment.

    Args:
        env_id: Environment ID

    Returns:
        Dictionary containing environment information
    """
    return {
        "action_space": env.action_space,
        "observation_space": env.observation_space,
        "reward_config": env.reward_config,
        "metadata": env.metadata,
        "spec": env.spec,
    }


def validate_action(env: ChromeDinoEnv, action: Any) -> bool:
    """Validate if an action is valid for the environment.

    Args:
        env: Chrome Dino environment instance
        action: Action to validate

    Returns:
        True if action is valid, False otherwise
    """
    return env.action_space.contains(action)


def benchmark_env(
    env_id: str = "ChromeDino-v0", episodes: int = 10
) -> dict[str, float]:
    """
    Run a simple benchmark of the environment.

    Args:
        env_id: Environment ID
        episodes: Number of episodes to run

    Returns:
        Benchmark statistics
    """
    env = create_env(env_id)

    episode_lengths = []
    episode_rewards = []
    episode_scores = []

    for _ in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                episode_scores.append(info.get("score", 0))
                break

    env.close()

    return {
        "episodes": episodes,
        "avg_length": sum(episode_lengths) / len(episode_lengths),
        "avg_reward": sum(episode_rewards) / len(episode_rewards),
        "avg_score": sum(episode_scores) / len(episode_scores),
        "max_length": max(episode_lengths),
        "max_reward": max(episode_rewards),
        "max_score": max(episode_scores),
    }
