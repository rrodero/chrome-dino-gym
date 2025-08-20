"""Human player interface for Chrome Dino game."""

import pygame

import chrome_dino_gym


def human_play():
    """Allow human to play the Chrome Dino game."""
    print("Chrome Dino - Human Player")
    print("Controls:")
    print("  SPACE/UP: Jump")
    print("  DOWN: Duck")
    print("  ESC/Q: Quit")
    print("  R: Restart after game over")
    print("\nPress any key in the game window to start...")

    # Create environment with human rendering
    env = chrome_dino_gym.make("ChromeDino-v0", render_mode="human")

    # Initialize pygame for input handling
    pygame.init()
    clock = pygame.time.Clock()

    # Game state
    running = True
    obs, info = env.reset()

    try:
        while running:
            # Handle pygame events
            action = 0  # Default: idle

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        running = False
                    elif event.key in [pygame.K_SPACE, pygame.K_UP]:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_r:
                        # Restart game
                        obs, info = env.reset()
                        print("Game restarted!")
                        continue

            # Check for continuous key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:
                action = 2  # Hold duck

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Render game
            env.render()

            # Handle game over
            if terminated or truncated:
                print("\nGame Over!")
                print(f"Final Score: {info['score']}")
                print(f"Obstacles Passed: {info['obstacles_passed']}")
                print("Press R to restart or ESC to quit")

                # Wait for restart or quit
                waiting = True
                while waiting and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                obs, info = env.reset()
                                waiting = False
                                print("Game restarted!")

                    clock.tick(10)  # Lower FPS while waiting

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env.close()
        pygame.quit()
        print("Thanks for playing!")


if __name__ == "__main__":
    human_play()
