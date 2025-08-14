"""Core game engine for Chrome Dino game."""

import random
from typing import Any

from .game_objects import Bird, Cactus, Cloud, DinoAction, Dinosaur, Obstacle
from .physics import PHYSICS


class DinoGameEngine:
    """
    Core game engine handling game logic, physics, and state management.

    This class manages:
    - Game state (score, speed, game over status)
    - Game objects (dinosaur, obstacles, clouds)
    - Obstacle spawning and management
    - Collision detection
    - Physics updates
    """

    def __init__(self, width: int = None, height: int = None):
        """Initialize game engine with optional custom dimensions."""
        self.width = width or PHYSICS.GAME_WIDTH
        self.height = height or PHYSICS.GAME_HEIGHT
        self.ground_y = PHYSICS.GROUND_Y

        # Game State
        self.score = 0
        self.game_over = False
        self.speed = PHYSICS.BASE_SPEED

        # Game Objects
        self.dinosaur = Dinosaur(x=PHYSICS.DINO_X_POSITION, y=self.ground_y)
        self.obstacles: list[Obstacle] = []
        self.clouds: list[Cloud] = []

        # Obstacle spawning
        self.obstacle_timer = 0
        self.next_obstacle_distance = random.randint(
            PHYSICS.MIN_OBSTACLE_GAP, PHYSICS.MAX_OBSTACLE_GAP
        )

        # Cloud spawning (decorative)
        self.cloud_timer = 0
        self.cloud_spawn_rate = random.randint(
            PHYSICS.CLOUD_SPAWN_RATE_MIN, PHYSICS.CLOUD_SPAWN_RATE_MAX
        )

    def reset(self):
        """Reset game to initial state."""
        self.score = 0
        self.game_over = False
        self.speed = PHYSICS.BASE_SPEED
        self.dinosaur = Dinosaur(x=PHYSICS.DINO_X_POSITION, y=self.ground_y)
        self.obstacles.clear()
        self.clouds.clear()
        self.obstacle_timer = 0
        self.cloud_timer = 0
        self.next_obstacle_distance = random.randint(
            PHYSICS.MIN_OBSTACLE_GAP, PHYSICS.MAX_OBSTACLE_GAP
        )

    def update(self, action: DinoAction):
        """Update game state based on player action."""
        if self.game_over:
            return

        # Handle dinosaur action
        if action == DinoAction.JUMP:
            self.dinosaur.jump()
        elif action == DinoAction.DUCK:
            self.dinosaur.duck()
        else:
            self.dinosaur.stop_duck()

        # Update dinosaur physics
        self.dinosaur.update()

        # Update obstacles
        self._update_obstacles()

        # Update clouds
        self._update_clouds()

        # Check collisions
        self._check_collisions()

        # Update game progression
        self._update_game_progression()

    def _update_obstacles(self):
        """Update obstacle positions and spawn new ones."""
        # Update existing obstacles
        for obstacle in self.obstacles:
            obstacle.update()
            # Remove obstacles that are off-screen
            if obstacle.is_off_screen():
                self.obstacles.remove(obstacle)

        # Spawn new obstacles
        self.obstacle_timer += self.speed
        if self.obstacle_timer >= self.next_obstacle_distance:
            self._spawn_obstacle()
            self.obstacle_timer = 0
            self.next_obstacle_distance = random.randint(
                PHYSICS.MIN_OBSTACLE_GAP, PHYSICS.MAX_OBSTACLE_GAP
            )

    def _spawn_obstacle(self):
        """Spawn a new obstacle at the right edge of screen."""
        obstacle_type = random.choice(["cactus", "bird"])

        if obstacle_type == "cactus":
            variant = random.randint(0, len(PHYSICS.CACTUS_VARIANTS) - 1)
            obstacle = Cactus(x=self.width, variant=variant, speed=self.speed)
        else:  # bird
            flight_level = random.randint(0, len(PHYSICS.BIRD_FLIGHT_HEIGHTS) - 1)
            obstacle = Bird(x=self.width, flight_level=flight_level, speed=self.speed)

        self.obstacles.append(obstacle)

    def _update_clouds(self):
        """Update decorative clouds."""
        # Move existing clouds
        for cloud in self.clouds:
            cloud.update()
            if cloud.is_off_screen():
                self.clouds.remove(cloud)

        # Spawn new clouds
        self.cloud_timer += 1
        if self.cloud_timer >= self.cloud_spawn_rate:
            self.clouds.append(Cloud(x=self.width, y=random.randint(20, 100)))
            self.cloud_timer = 0
            self.cloud_spawn_rate = random.randint(
                PHYSICS.CLOUD_SPAWN_RATE_MIN, PHYSICS.CLOUD_SPAWN_RATE_MAX
            )

    def _check_collisions(self):
        """Check for collisions between dinosaur and obstacles."""
        for obstacle in self.obstacles:
            if self.dinosaur.collides_with(obstacle):
                self.game_over = True
                break

    def _update_game_progression(self):
        """Update score and speed progression."""
        self.score += 1

        # Increase speed gradually
        if self.speed < PHYSICS.MAX_SPEED:
            self.speed += PHYSICS.SPEED_INCREMENT
            # Update all obstacle speeds
            for obstacle in self.obstacles:
                obstacle.speed = self.speed

    def get_obstacles_passed(self) -> int:
        """Get number of obstacles that have passed the dinosaur."""
        return len(
            [obs for obs in self.obstacles if obs.x + obs.width < self.dinosaur.x]
        )

    def get_upcoming_obstacles(self, count: int = 3) -> list[Obstacle]:
        """Get list of upcoming obstacles sorted by distance."""
        upcoming = [obs for obs in self.obstacles if obs.x > self.dinosaur.x]
        upcoming.sort(key=lambda obs: obs.x)
        return upcoming[:count]

    def get_state(self) -> dict[str, Any]:
        """Get current game state for observation."""
        # Find closest obstacles
        upcoming_obstacles = self.get_upcoming_obstacles(3)

        # Get info about next few obstacles
        obstacle_data = []
        for obstacle in upcoming_obstacles:
            obstacle_data.extend(
                [
                    (obstacle.x - self.dinosaur.x) / self.width,  # Normalized distance
                    obstacle.y / self.height,  # Normalized height
                    obstacle.width / 100.0,  # Normalized width
                    obstacle.height / 100.0,  # Normalized height
                    1.0 if obstacle.obstacle_type == "cactus" else 0.0,  # Type encoding
                ]
            )

        # Pad with zeros if fewer than 3 obstacles
        while len(obstacle_data) < 15:  # 3 obstacles * 5 features each
            obstacle_data.append(0.0)

        # Dinosaur state
        dino_state = [
            self.dinosaur.y / self.height,  # Normalized height
            self.dinosaur.velocity_y / 20.0,  # Normalized velocity
            1.0 if self.dinosaur.is_jumping else 0.0,  # Jump state
            1.0 if self.dinosaur.is_ducking else 0.0,  # Duck state
            self.speed / PHYSICS.MAX_SPEED,  # Normalized speed
        ]

        # Combine all state information
        state_vector = dino_state + obstacle_data

        return {
            "vector": state_vector,
            "score": self.score,
            "speed": self.speed,
            "game_over": self.game_over,
            "dinosaur": self.dinosaur.get_state_dict(),
            "obstacles": len(self.obstacles),
            "upcoming_obstacles": len(upcoming_obstacles),
        }
