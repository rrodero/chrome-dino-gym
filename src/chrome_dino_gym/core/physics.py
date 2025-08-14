"""Physics constants and configuration for the game."""

from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    """Configuration for game physics."""

    # Gravity and movement
    GRAVITY: float = 0.6
    JUMP_VELOCITY: float = -12.0
    BASE_SPEED: float = 6.0
    SPEED_INCREMENT: float = 0.001
    MAX_SPEED: float = 15.0

    # Ground and boundaries
    GROUND_Y: float = 200.0
    GAME_WIDTH: int = 600
    GAME_HEIGHT: int = 300

    # Dinosaur dimensions
    DINO_NORMAL_WIDTH: int = 44
    DINO_NORMAL_HEIGHT: int = 47
    DINO_DUCK_WIDTH: int = 59
    DINO_DUCK_HEIGHT: int = 26
    DINO_X_POSITION: float = 50.0

    # Obstacle spawning
    MIN_OBSTACLE_GAP: int = 120
    MAX_OBSTACLE_GAP: int = 200

    # Cactus variants (width, height)
    CACTUS_VARIANTS: list[tuple[int, int]] = field(default_factory=list)

    # Bird flight heights
    BIRD_FLIGHT_HEIGHTS: list[float] = field(default_factory=list)
    BIRD_WIDTH: int = 46
    BIRD_HEIGHT: int = 40

    # Cloud spawning (decorative)
    CLOUD_SPAWN_RATE_MIN: int = 200
    CLOUD_SPAWN_RATE_MAX: int = 400
    CLOUD_SPEED: float = 1.0

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if len(self.CACTUS_VARIANTS) == 0:
            self.CACTUS_VARIANTS = [(17, 35), (34, 35), (51, 35)]

        if len(self.BIRD_FLIGHT_HEIGHTS) == 0:
            self.BIRD_FLIGHT_HEIGHTS = [150, 100, 75]


PHYSICS = PhysicsConfig()
