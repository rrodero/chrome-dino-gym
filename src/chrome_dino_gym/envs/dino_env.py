"""Chrome Dino Gymnasium Environment implementation."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core import DinoAction, DinoGameEngine
from ..rendering import PyGameRenderer


class ChromeDinoEnv(gym.Env):
    """
    Chrome Dino Game Gymnasium Environment.

    A reinforcement learning environment based on the Chrome Dino game.
    The agent controls a dinosaur that must avoid obstacles by jumping or ducking.

    Action Space:
        Discrete(3):
        - 0: Do nothing (idle)
        - 1: Jump
        - 2: Duck

    Observation Space:
        Box(20,) containing:
        - Dinosaur state (position, velocity, action states)
        - Game speed information
        - Information about upcoming obstacles

    Reward:
        - +0.1 for each step survived
        - +10.0 for each obstacle successfully passed
        - -100.0 for collision (game over)

    Episode Termination:
        - When dinosaur collides with an obstacle
        - When maximum episode length is reached (if specified)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
        width: int = 600,
        height: int = 300,
        reward_config: dict[str, float] | None = None,
    ):
        """
        Initialize Chrome Dino Environment.

        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            max_episode_steps: Maximum steps per episode (None for unlimited)
            width: Game window width
            height: Game window height
            reward_config: Custom reward configuration
        """

        super().__init__()

        # Environment configuration
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.width = width
        self.height = height

        # Reward configuration
        self.reward_config = {
            "step_reward": 0.1,
            "obstacle_reward": 10.0,
            "collision_penalty": -100.0,
        }
        if reward_config:
            self.reward_config.update(reward_config)

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(20,), dtype=np.float32
        )

        # Game components
        self.game = DinoGameEngine(width=width, height=height)
        self.renderer = None

        # Episode tracking
        self.step_count = 0
        self.last_obstacles_passed = 0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset game state
        self.game.reset()
        self.step_count = 0
        self.last_obstacles_passed = 0

        # Get initial observation
        state = self.game.get_state()
        observation = np.array(state["vector"], dtype=np.float32)

        info = {
            "score": state["score"],
            "speed": state["speed"],
            "obstacles_passed": 0,
            "step_count": self.step_count,
        }

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step."""
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Convert action to enum
        dino_action = DinoAction(action)

        # Store previous state for reward calculation
        prev_obstacles_passed = self.game.get_obstacles_passed()

        # Update game
        self.game.update(dino_action)
        self.step_count += 1

        # Get new state
        state = self.game.get_state()
        observation = np.array(state["vector"], dtype=np.float32)

        # Calculate reward
        reward = self._calculate_reward(state, prev_obstacles_passed)

        # Check termination conditions
        terminated = state["game_over"]
        truncated = (
            self.max_episode_steps is not None
            and self.step_count >= self.max_episode_steps
        )

        # Update tracking
        current_obstacles_passed = self.game.get_obstacles_passed()

        # Prepare info
        info = {
            "score": state["score"],
            "speed": state["speed"],
            "obstacles_passed": current_obstacles_passed,
            "step_count": self.step_count,
            "game_over_reason": "collision" if terminated else None,
        }

        return observation, reward, terminated, truncated, info

    def _calculate_reward(
        self, state: dict[str, Any], prev_obstacles_passed: int
    ) -> float:
        """Calculate reward based on current state and previous state."""
        reward = 0.0

        if state["game_over"]:
            # Large penalty for collision
            reward += self.reward_config["collision_penalty"]
        else:
            # Small reward for surviving each step
            reward += self.reward_config["step_reward"]

            # Bonus reward for passing obstacles
            current_obstacles_passed = self.game.get_obstacles_passed()
            new_obstacles_passed = current_obstacles_passed - prev_obstacles_passed
            if new_obstacles_passed > 0:
                reward += self.reward_config["obstacle_reward"] * new_obstacles_passed

        return reward

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode is None:
            return None

        # Initialize renderer if needed
        if self.renderer is None:
            self.renderer = PyGameRenderer(
                width=self.width, height=self.height, render_mode=self.render_mode
            )
            return self.renderer.render(self.game)

        return None

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
