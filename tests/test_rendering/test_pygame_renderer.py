import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Import the class to be tested and its dependencies
from src.chrome_dino_gym.rendering.pygame_renderer import PyGameRenderer

# Mocking the Pygame and core modules
# This is done before importing the class under test
pygame = MagicMock()
sys.modules["pygame"] = pygame
sys.modules["pygame.locals"] = MagicMock()
sys.modules["pygame.display"] = MagicMock()
sys.modules["pygame.font"] = MagicMock()
sys.modules["pygame.surfarray"] = MagicMock()
pygame.init.return_value = (1, 0)
pygame.font.init.return_value = None
pygame.font.get_default_font.return_value = "Arial"


# Define dummy classes for mocking game objects
class MockDinosaur:
    def __init__(
        self, x=0, y=0, width=20, height=20, is_jumping=False, is_ducking=False
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_jumping = is_jumping
        self.is_ducking = is_ducking


class MockCactus:
    def __init__(self, x=0, y=0, width=20, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class MockBird:
    def __init__(self, x=0, y=0, width=20, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class MockCloud:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class MockDinoGameEngine:
    def __init__(self, score=0, speed=1.0, game_over=False):
        self.score = score
        self.speed = speed
        self.game_over = game_over
        self.dinosaur = MockDinosaur()
        self.obstacles = []
        self.clouds = []
        self.ground_y = 200


# Fixtures for common setup
@pytest.fixture
def mock_pygame_modules():
    """Fixture to ensure pygame mocks are reset for each test."""
    pygame.reset_mock()
    pygame.init.return_value = (1, 0)
    pygame.display.set_mode.return_value = MagicMock()
    pygame.Surface.return_value = MagicMock()
    pygame.font.Font.return_value.render.return_value = MagicMock()
    pygame.surfarray.array3d.return_value = np.zeros((600, 300, 3), dtype=np.uint8)


@pytest.fixture
def renderer_human(mock_pygame_modules):
    """Fixture to create a renderer in 'human' mode."""
    return PyGameRenderer(width=600, height=300, render_mode="human")


@pytest.fixture
def renderer_rgb_array(mock_pygame_modules):
    """Fixture to create a renderer in 'rgb_array' mode."""
    return PyGameRenderer(width=600, height=300, render_mode="rgb_array")


@pytest.fixture
def mock_game_engine():
    """Fixture to create a mock game engine instance."""
    return MockDinoGameEngine()


# The actual test functions
def test_init_human_mode(renderer_human):
    """Test initialization in 'human' render mode."""
    assert renderer_human.is_initialized
    pygame.display.set_mode.assert_called_once_with((600, 300))
    assert renderer_human.screen is not None
    assert renderer_human.display_surface == renderer_human.screen


def test_init_rgb_array_mode(renderer_rgb_array):
    """Test initialization in 'rgb_array' render mode."""
    assert renderer_rgb_array.is_initialized
    pygame.display.set_mode.assert_not_called()
    assert renderer_rgb_array.screen is not None
    assert renderer_rgb_array.display_surface is None


def test_render_human_mode(renderer_human, mock_game_engine):
    """Test the render method for 'human' mode."""
    result = renderer_human.render(mock_game_engine)
    assert result is None
    pygame.display.flip.assert_called_once()
    renderer_human.clock.tick.assert_called_once_with(60)


def test_render_rgb_array_mode(renderer_rgb_array, mock_game_engine):
    """Test the render method for 'rgb_array' mode."""
    mock_surfarray_3d = np.zeros((600, 300, 3), dtype=np.uint8)
    pygame.surfarray.array3d.return_value = mock_surfarray_3d

    result = renderer_rgb_array.render(mock_game_engine)

    assert isinstance(result, np.ndarray)
    assert result.shape == (300, 600, 3)
    assert result.dtype == np.uint8


def test_draw_dinosaur_not_game_over(renderer_human, mock_game_engine):
    """Test that the dinosaur is drawn correctly when the game is not over."""
    renderer_human._draw_dinosaur(mock_game_engine.dinosaur, game_over=False)
    assert pygame.draw.rect.called
    assert pygame.draw.circle.called


def test_draw_dinosaur_game_over(renderer_human):
    """Test that the dinosaur is drawn with a 'dead' color when the game is over."""
    dinosaur = MockDinosaur()
    renderer_human._draw_dinosaur(dinosaur, game_over=True)
    draw_calls = [
        call
        for call in pygame.draw.rect.call_args_list
        if call[0][1] == renderer_human.colors["dinosaur_dead"]
    ]
    assert len(draw_calls) > 0


def test_draw_ui_game_over(renderer_human, mock_game_engine, mocker):
    """Test that the 'GAME OVER' message is drawn when the game is over."""
    mock_game_engine.game_over = True
    renderer_human.font = MagicMock()
    renderer_human._draw_ui(mock_game_engine)
    args_list_str = str(renderer_human.font.render.call_args_list)
    assert "GAME OVER" in args_list_str
    assert "Press R to restart" in args_list_str


def test_close_method(renderer_human):
    """Test that the close method calls pygame.quit."""
    assert renderer_human.is_initialized
    renderer_human.close()
    pygame.display.quit.assert_called_once()
    pygame.quit.assert_called_once()
    assert not renderer_human.is_initialized


def test_draw_obstacles(renderer_human, mock_game_engine, mocker):
    """Test that different types of obstacles are drawn correctly."""
    mock_game_engine.obstacles = [MockCactus(), MockBird()]

    mocker.patch.object(renderer_human, "_draw_cactus")
    mocker.patch.object(renderer_human, "_draw_bird")

    renderer_human.render(mock_game_engine)

    renderer_human._draw_cactus.assert_called_once_with(mock_game_engine.obstacles[0])
    renderer_human._draw_bird.assert_called_once_with(mock_game_engine.obstacles[1])
