"""Command line interface for Chrome Dino Gym."""

import argparse
import time

from .utils import benchmark_env, create_env


def demo(render_mode: str = "human", episodes: int = 3) -> None:
    """Run a demo of the environment with random actions."""
    print("Starting Chrome Dino Demo...")
    print("Press Ctrl+C to stop early")

    env = create_env(render_mode=render_mode)

    try:
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            obs, info = env.reset()

            total_reward = 0
            step_count = 0

            while True:
                # Random action
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                if render_mode == "human":
                    env.render()
                    time.sleep(0.016)

                if terminated or truncated:
                    print(f"Episode ended after {step_count} steps")
                    print(f"Total reward: {total_reward:.2f}")
                    print(f"Final score: {info['score']}")
                    print(f"Obstacles passed: {info['obstacles_passed']}")
                    break

            if episode < episodes - 1:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()
        print("Demo completed!")


def train():
    """Placeholder for training command."""
    print("Training functionality not implemented yet.")
    print("This would typically involve:")
    print("1. Loading a training configuration")
    print("2. Creating the environment")
    print("3. Initializing an RL agent (e.g., DQN, PPO)")
    print("4. Running the training loop")
    print("5. Saving the trained model")
    print("\nSee examples/ directory for training implementations.")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Chrome Dino Gym CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run environment demo")
    demo_parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Rendering mode",
    )
    demo_parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run environment benchmark")
    bench_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes for benchmark"
    )
    bench_parser.add_argument(
        "--env-id", default="ChromeDino-v0", help="Environment ID to benchmark"
    )

    # Train command
    subparsers.add_parser("train", help="Train an RL agent")

    args = parser.parse_args()

    if args.command == "demo":
        demo(render_mode=args.render_mode, episodes=args.episodes)
    elif args.command == "benchmark":
        print(f"Benchmarking {args.env_id} for {args.episodes} episodes...")
        results = benchmark_env(args.env_id, args.episodes)
        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    elif args.command == "train":
        train()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
