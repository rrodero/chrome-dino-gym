"""Tests for the game engine."""

from chrome_dino_gym.core import DinoAction, DinoGameEngine
from chrome_dino_gym.core.physics import PHYSICS


class TestDinoGameEngine:
    """Test cases for DinoGameEngine."""

    def test_initialization(self):
        """Test game engine initialization."""
        engine = DinoGameEngine()

        assert engine.score == 0
        assert not engine.game_over
        assert engine.speed == PHYSICS.BASE_SPEED
        assert len(engine.obstacles) == 0
        assert len(engine.clouds) == 0

    def test_reset(self):
        """Test game engine reset functionality."""
        engine = DinoGameEngine()

        # Modify state
        engine.score = 100
        engine.game_over = True
        engine.speed = 10.0
        engine.obstacles.append("dummy_obstacle")

        # Reset
        engine.reset()

        # Check reset state
        assert engine.score == 0
        assert not engine.game_over
        assert engine.speed == PHYSICS.BASE_SPEED
        assert len(engine.obstacles) == 0
        assert len(engine.clouds) == 0

    def test_update_with_actions(self):
        """Test game updates with different actions."""
        engine = DinoGameEngine()

        # Test jump action
        initial_y = engine.dinosaur.y
        engine.update(DinoAction.JUMP)

        # Dinosaur should be jumping after first update
        assert engine.dinosaur.is_jumping

        # Update physics - dinosaur should move up initially
        engine.dinosaur.update()
        assert engine.dinosaur.y < initial_y

        # Test duck action (reset first)
        engine.reset()
        engine.update(DinoAction.DUCK)
        assert engine.dinosaur.is_ducking

        # Test idle action
        engine.reset()
        engine.update(DinoAction.IDLE)
        assert not engine.dinosaur.is_jumping
        assert not engine.dinosaur.is_ducking

    def test_score_progression(self):
        """Test score increases over time."""
        engine = DinoGameEngine()
        initial_score = engine.score

        # Update game state multiple times
        for _ in range(10):
            engine.update(DinoAction.IDLE)

        assert engine.score > initial_score

    def test_speed_progression(self):
        """Test speed increases over time."""
        engine = DinoGameEngine()
        initial_speed = engine.speed

        # Update game state many times
        for _ in range(1000):
            if engine.game_over:
                break
            engine.update(DinoAction.IDLE)

        assert engine.speed >= initial_speed

    def test_obstacle_spawning(self):
        """Test that obstacles are spawned over time."""
        engine = DinoGameEngine()

        # Run long enough for obstacles to spawn
        for _ in range(500):
            if engine.game_over:
                break
            engine.update(DinoAction.IDLE)

        # Should have spawned some obstacles
        assert len(engine.obstacles) > 0

    def test_get_state(self):
        """Test state retrieval."""
        engine = DinoGameEngine()
        state = engine.get_state()

        assert isinstance(state, dict)
        assert "vector" in state
        assert "score" in state
        assert "speed" in state
        assert "game_over" in state
        assert len(state["vector"]) == 20
