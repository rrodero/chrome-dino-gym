"""Game objects including dinosaur, obstacles, and other entities."""

from dataclasses import dataclass
from enum import Enum

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

    def __post_init__(self):
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

    def stop_duck(self):
        """Stop ducking."""
        if self.is_ducking:
            self.is_ducking = False
            self._update_dimensions()

    def _update_dimensions(self):
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

    def update(self):
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

    def get_state_dict(self) -> dict:
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
