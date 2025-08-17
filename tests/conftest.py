"""Pytest configuration and fixtures"""

import pytest

from chrome_dino_gym import ChromeDinoEnv


@pytest.fixture
def env():
    """Create a basic environment for testing"""
    env = ChromeDinoEnv(render_mode=None, max_episode_steps=100)
    yield env
    env.close()


@pytest.fixture
def env_with_render():
    """Create an environment with rgb_array rendering"""
    env = ChromeDinoEnv(render_mode="rgb_array", max_episode_steps=100)
    yield env
    env.close()


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests"""
    return 42
