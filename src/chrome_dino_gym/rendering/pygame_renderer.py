"""Pygame-based renderer for Chrome Dino game."""

import numpy as np
import pygame

from ..core import DinoGameEngine
from ..core.game_objects import Bird, Cactus, Dinosaur


class PyGameRenderer:
    """
    Pygame-based renderer for the Chrome Dino game.

    Handles visual rendering with support for both human-viewable display
    and rgb_array output for machine learning applications.
    """

    def __init__(self, width: int, height: int, render_mode: str = "human"):
        """
        Initialize pygame renderer.

        Args:
            width: Screen width
            height: Screen height
            render_mode: Either "human" or "rgb_array"
        """
        self.width = width
        self.height - height
        self.render_mode = render_mode

        # Initialize pygame
        pygame.init()
        pygame.display.init()

        # Create surface
        if render_mode == "human":
            pygame.display.set_mode((width, height))
            pygame.display.set_caption("AI Chrome Dino Game")
            self.clock = pygame.time.Clock()

        self.screen = pygame.Surface((width, height))

        # Colors
        self.colors = {
            "background": (247, 247, 247),  # Light gray
            "ground": (83, 83, 83),  # Dark gray
            "dinosaur": (83, 83, 83),  # Dark gray
            "dinosaur_dead": (255, 0, 0),  # Red
            "cactus": (0, 150, 0),  # Green
            "bird": (100, 100, 100),  # Gray
            "cloud": (247, 247, 247),  # Light gray
            "text": (83, 83, 83),  # Dark gray
            "white": (255, 255, 255),  # White
            "black": (0, 0, 0),  # Black
        }

        # Initialize font
        self.font = pygame.font.Font(None, 24)

    def render(self, game: DinoGameEngine) -> np.ndarray | None:
        """
        Render the current game state.

        Args:
            game: Game engine instance to render

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """

        # Clear screen
        self.screen.fill(self.colors["background"])

        # Draw ground line
        ground_y = int(game.ground_y + game.dinosaur.height)
        pygame.draw.line(
            self.screen, self.colors["ground"], (0, ground_y), (self.width, ground_y), 2
        )

        # Draw clouds
        for cloud in game.clouds:
            self._draw_cloud(cloud.x, cloud.y)

        # Draw dinosaur
        self._draw_dinosaur(game.dinosaur, game.game_over)

        # Draw obstacles
        for obstacle in game.obstacles:
            self._draw_obstacle(obstacle)

        # Draw UI elements
        self._draw_ui(game)

        if self.render_mode == "human":
            pygame.display.flip()
            if hasattr(self, "clock"):
                self.clock.tick(60)
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def _draw_dinosaur(self, dinosaur: Dinosaur, game_over: bool) -> None:
        """Draw the dinosaur character."""
        color = self.colors["dinosaur_dead"] if game_over else self.colors["dinosaur"]

        # Main body rectangle
        pygame.draw.rect(
            self.screen,
            color,
            (int(dinosaur.x), int(dinosaur.y), dinosaur.width, dinosaur.height),
        )

        # Add simple details if not ducking
        if not dinosaur.is_ducking:
            # Eye
            eye_x = int(dinosaur.x + dinosaur.width * 0.7)
            eye_y = int(dinosaur.y + dinosaur.height * 0.25)
            pygame.draw.circle(self.screen, self.colors["white"], (eye_x, eye_y), 4)
            pygame.draw.circle(self.screen, self.colors["black"], (eye_x + 1, eye_y), 2)

            # Simple legs (lines)
            leg_y = int(dinosaur.y + dinosaur.height)
            leg1_x = int(dinosaur.x + dinosaur.width * 0.3)
            leg2_x = int(dinosaur.x + dinosaur.width * 0.6)

            if not dinosaur.is_jumping:
                pygame.draw.line(
                    self.screen, color, (leg1_x, leg_y), (leg1_x, leg_y + 5), 3
                )
                pygame.draw.line(
                    self.screen, color, (leg2_x, leg_y), (leg2_x, leg_y + 5), 3
                )

    def _draw_obstacle(self, obstacle) -> None:
        """Draw an obstacle (cactus or bird)."""
        if isinstance(obstacle, Cactus):
            self._draw_cactus(obstacle)
        elif isinstance(obstacle, Bird):
            self._draw_bird(obstacle)
        else:
            # Generic obstacle
            pygame.draw.rect(
                self.screen,
                self.colors["cactus"],
                (int(obstacle.x), int(obstacle.y), obstacle.width, obstacle.height),
            )

    def _draw_cactus(self, cactus: Cactus) -> None:
        """Draw a cactus obstacle with some detail."""
        # Main body
        pygame.draw.rect(
            self.screen,
            self.colors["cactus"],
            (int(cactus.x), int(cactus.y), cactus.width, cactus.height),
        )

        # Add simple cactus arms for larger variants
        if cactus.width > 20:
            arm_y = int(cactus.y + cactus.height * 0.4)
            arm_width = max(3, cactus.width // 8)

            # Left arm
            pygame.draw.rect(
                self.screen,
                self.colors["cactus"],
                (int(cactus.x - arm_width), arm_y, arm_width, cactus.height // 3),
            )

            # Right arm (for very wide cacti)
            if cactus.width > 40:
                pygame.draw.rect(
                    self.screen,
                    self.colors["cactus"],
                    (
                        int(cactus.x + cactus.width),
                        arm_y,
                        arm_width,
                        cactus.height // 3,
                    ),
                )

    def _draw_bird(self, bird: Bird) -> None:
        """Draw a bird obstacle."""
        # Main body (ellipse)
        pygame.draw.ellipse(
            self.screen,
            self.colors["bird"],
            (int(bird.x), int(bird.y), bird.width, bird.height),
        )

        # Simple wing details
        wing_y = int(bird.y + bird.height * 0.3)
        pygame.draw.arc(
            self.screen,
            self.colors["black"],
            (int(bird.x + 5), wing_y, bird.width - 10, bird.height // 2),
            0,
            3.14,
            2,
        )

    def _draw_cloud(self, x: float, y: float) -> None:
        """Draw a simple cloud."""
        # Cloud as overlapping circles
        cloud_parts = [(0, 0, 12), (15, 0, 10), (25, 0, 8), (8, -8, 8), (20, -6, 6)]

        for dx, dy, radius in cloud_parts:
            pygame.draw.circle(
                self.screen, self.colors["cloud"], (int(x + dx), int(y + dy)), radius
            )

    def _draw_ui(self, game: DinoGameEngine) -> None:
        """Draw UI elements like score and speed."""
        if self.font is None:
            return

        try:
            # Score
            score_text = self.font.render(
                f"Score: {game.score}", True, self.colors["text"]
            )
            self.screen.blit(score_text, (self.width - 120, 20))

            # Speed
            speed_text = self.font.render(
                f"Speed: {game.speed:.1f}", True, self.colors["text"]
            )
            self.screen.blit(speed_text, (self.width - 120, 45))

            # Game Over message
            if game.game_over:
                game_over_text = self.font.render(
                    "GAME OVER", True, self.colors["dinosaur_dead"]
                )
                text_rect = game_over_text.get_rect(
                    center=(self.width // 2, self.height // 2)
                )
                self.screen.blit(game_over_text, text_rect)

                restart_text = self.font.render(
                    "Press R to restart", True, self.colors["text"]
                )
                restart_rect = restart_text.get_rect(
                    center=(self.width // 2, self.height // 2 + 30)
                )
                self.screen.blit(restart_text, restart_rect)

        except Exception:
            # Fallback if text rendering fails
            pass

    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.display.quit()
        pygame.quit()
