"""Game objects including dinosaur, obstacles, and other entities."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pygame

from .physics import PHYSICS


class DinoAction(Enum):
    """Available actions for the dinosaur."""

    IDLE = 0
    JUMP = 1
    DUCK = 2


@dataclass
class GameObject:
    """Base class for game objects with position and dimensions."""

    x: float
    y: float
    width: int
    height: int

    def get_rect(self) -> pygame.Rect:
        """Get pygame rectangle for collision detection."""
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def collides_with(self, other: "GameObject") -> bool:
        """Check collision with another game object."""
        return self.get_rect().colliderect(other.get_rect())


@dataclass
class Dinosaur(GameObject):
    """Dinosaur player character with physics and state management."""

    velocity_y: float = 0.0
    is_jumping: bool = False
    is_ducking: bool = False

    def __post_init__(self) -> None:
        """Initialize dinosaur at ground position."""
        self.y = PHYSICS.GROUND_Y
        self.width = PHYSICS.DINO_NORMAL_WIDTH
        self.height = PHYSICS.DINO_NORMAL_HEIGHT

    def jump(self) -> bool:
        """Make dinosaur jump if on ground. Returns True if jump was executed."""
        if not self.is_jumping:
            self.velocity_y = PHYSICS.JUMP_VELOCITY
            self.is_jumping = True
            self.is_ducking = False
            self._update_dimensions()
            return True
        return False

    def duck(self) -> bool:
        """Make dinosaur duck if on ground. Returns True if duck was executed."""
        if not self.is_jumping:
            self.is_ducking = True
            self._update_dimensions()
            return True
        return False

    def stop_duck(self) -> None:
        """Stop ducking."""
        if self.is_ducking:
            self.is_ducking = False
            self._update_dimensions()

    def _update_dimensions(self) -> None:
        """Update dinosaur dimensions based on state."""
        if self.is_ducking and not self.is_jumping:
            old_height = self.height
            self.width = PHYSICS.DINO_DUCK_WIDTH
            self.height = PHYSICS.DINO_DUCK_HEIGHT
            # Adjust y position to keep dinosaur on ground
            self.y += old_height - self.height
        else:
            if self.height != PHYSICS.DINO_NORMAL_HEIGHT:  # Was ducking
                self.y -= PHYSICS.DINO_NORMAL_HEIGHT - self.height
            self.width = PHYSICS.DINO_NORMAL_WIDTH
            self.height = PHYSICS.DINO_NORMAL_HEIGHT

    def update(self) -> None:
        """Update dinosaur physics and position."""
        if self.is_jumping:
            self.velocity_y += PHYSICS.GRAVITY
            self.y += self.velocity_y

            # Check if landed
            if self.y >= PHYSICS.GROUND_Y:
                self.y = PHYSICS.GROUND_Y
                self.velocity_y = 0.0
                self.is_jumping = False
                # Update dimensions in case we were ducking while jumping
                self._update_dimensions()

    def get_state_dict(self) -> dict[str, Any]:
        """Get dinosaur state as dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "velocity_y": self.velocity_y,
            "is_jumping": self.is_jumping,
            "is_ducking": self.is_ducking,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Obstacle(GameObject):
    """Base obstacle class."""

    speed: float = PHYSICS.BASE_SPEED
    obstacle_type: str = "generic"

    def update(self) -> None:
        """Move obstacle leftward."""
        self.x -= self.speed

    def is_off_screen(self) -> bool:
        """Check if obstacle is completely off the left side of screen."""
        return self.x + self.width < 0


@dataclass
class Cactus(Obstacle):
    """Cactus obstacle with multiple variants."""

    obstacle_type: str = "cactus"
    variant: int = 0

    def __init__(self, x: float, variant: int = 0, speed: float | None = None):
        """Initialize cactus with specified variant."""
        self.variant = variant % len(PHYSICS.CACTUS_VARIANTS)
        variant_data = PHYSICS.CACTUS_VARIANTS[self.variant]

        super().__init__(
            x=x,
            y=PHYSICS.GROUND_Y - variant_data[1],  # Adjust y to ground level
            width=variant_data[0],
            height=variant_data[1],
            speed=speed or PHYSICS.BASE_SPEED,
            obstacle_type="cactus",
        )


@dataclass
class Bird(Obstacle):
    """Flying bird obstacle with multiple flight levels."""

    obstacle_type: str = "bird"
    flight_level: int = 0

    def __init__(self, x: float, flight_level: int = 0, speed: float | None = None):
        """Initialize bird with specified flight level."""
        self.flight_level = flight_level % len(PHYSICS.BIRD_FLIGHT_HEIGHTS)
        height = PHYSICS.BIRD_FLIGHT_HEIGHTS[self.flight_level]

        super().__init__(
            x=x,
            y=height,
            width=PHYSICS.BIRD_WIDTH,
            height=PHYSICS.BIRD_HEIGHT,
            speed=speed or PHYSICS.BASE_SPEED,
            obstacle_type="bird",
        )


@dataclass
class Cloud:
    """Decorative cloud object."""

    x: float
    y: float
    speed: float = PHYSICS.CLOUD_SPEED

    def update(self) -> None:
        """Move cloud leftward."""
        self.x -= self.speed

    def is_off_screen(self) -> bool:
        """Check if cloud is off screen."""
        return self.x + 46 < 0  # Assuming cloud width of 46
