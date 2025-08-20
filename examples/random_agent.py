"""Random agent baseline for Chrome Dino game."""

import matplotlib.pyplot as plt
import numpy as np

import chrome_dino_gym


class RandomAgent:
    """Simple random agent that takes random actions."""

    def __init__(self, action_space):
        """Initialize random agent."""
        self.action_space = action_space
        self.rng = np.random.RandomState(42)

    def act(self, observation):
        """Choose radndom action"""
        return self.rng.choice(self.action_space.n)

    def reset(self):
        """Reset agent state (nothing to do for random agent)"""
        pass


def evaluate_random_agent(episodes=10, render=False):
    """Evaluate random agent performance."""
    print(f"Evaluating Random Agent over {episodes} episodes...")

    # Create environment
    render_mode = "human" if render else None
    env = chrome_dino_gym.make("ChromeDino-v0", render_mode=render_mode)

    # Create random agent
    agent = RandomAgent(env.action_space)

    # Tracking variables
    episode_scores = []
    episode_lengths = []
    episode_rewards = []
    obstacles_passed_list = []

    for episode in range(episodes):
        obs, info = env.reset()
        agent.reset()

        episode_reward = 0
        episode_length = 0

        while True:
            # Agent chooses action
            action = agent.act(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            if terminated or truncated:
                episode_scores.append(info["score"])
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                obstacles_passed_list.append(info["obstacles_passed"])

                print(
                    f"Episode {episode + 1}: Score={info['score']}, "
                    f"Length={episode_length}, Reward={episode_reward:.1f}, "
                    f"Obstacles={info['obstacles_passed']}"
                )
                break
    env.close()

    # Calculate statistics
    stats = {
        "mean_score": np.mean(episode_scores),
        "std_score": np.std(episode_scores),
        "max_score": np.max(episode_scores),
        "mean_length": np.mean(episode_lengths),
        "mean_reward": np.mean(episode_rewards),
        "mean_obstacles": np.mean(obstacles_passed_list),
    }

    # Print results
    print("\nRandom Agent Results:")
    print("-" * 30)
    print(f"Episodes: {episodes}")
    print(f"Mean Score: {stats['mean_score']:.1f} ± {stats['std_score']:.1f}")
    print(f"Max Score: {stats['max_score']}")
    print(f"Mean Episode Length: {stats['mean_length']:.1f}")
    print(f"Mean Reward: {stats['mean_reward']:.1f}")
    print(f"Mean Obstacles Passed: {stats['mean_obstacles']:.1f}")

    # Plot results
    if not render:  # Only plot if not rendering to avoid conflicts
        plot_results(episode_scores, episode_lengths, episode_rewards)

    return stats


def plot_results(scores, lengths, rewards):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Random Agent Performance")

    # Score over episodes
    axes[0, 0].plot(scores)
    axes[0, 0].set_title("Score per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].grid(True)

    # Episode length over episodes
    axes[0, 1].plot(lengths)
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True)

    # Score distribution
    axes[1, 0].hist(scores, bins=20, alpha=0.7)
    axes[1, 0].set_title("Score Distribution")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True)

    # Reward over episodes
    axes[1, 1].plot(rewards)
    axes[1, 1].set_title("Reward per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Total Reward")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Random Agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--render", action="store_true", help="Render during evaluation"
    )

    args = parser.parse_args()

    evaluate_random_agent(episodes=args.episodes, render=args.render)
