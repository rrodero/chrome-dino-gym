"""Basic usage example for Chrome Dino Gym."""

import chrome_dino_gym


def basic_usage_example():
    """Demonstrate basic environment usage."""
    print("Chrome Dino Gym - Basic Usage Example")
    print("=" * 40)

    # Create environment
    env = chrome_dino_gym.make("ChromeDino-v0", render_mode="human")

    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}")

    # Run for a few steps
    episode_reward = 0
    step_count = 0

    for step in range(100):
        # Take random action
        action = env.action_space.sample()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        # Print progress occasionally
        if step % 20 == 0:
            print(f"Step {step}: Score={info['score']}, Reward={reward:.2f}")

        # Render environment
        env.render()

        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended after {step_count} steps")
            print(f"Final score: {info['score']}")
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Obstacles passed: {info['obstacles_passed']}")
            break

    # Clean up
    env.close()
    print("\nBasic usage example completed!")

    if __name__ == "__main__":
        basic_usage_example()
