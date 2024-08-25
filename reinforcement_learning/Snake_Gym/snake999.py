import gym
from gym import spaces
import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()
pygame.font.init()

# Define constants
BROWN = (139, 69, 19)  # Brown color
SNAKE_COLOR = (0, 255, 0)  # Green color for the snake
FRUIT_COLOR = (255, 0, 0)  # Red color for the fruit
HEAD_COLOR = (0, 0, 255)  # Blue color for the head of the snake

# Custom Gym Environment
class SnakeEnvWithPenalty(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=500, height=500, rows=7, cols=7):
        super(SnakeEnvWithPenalty, self).__init__()
        
        # Set dimensions
        self.WIDTH = width
        self.HEIGHT = height
        self.ROWS = rows
        self.COLS = cols
        self.CELL_SIZE = self.WIDTH // self.COLS  # Size of each cell
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.observation_space = spaces.Box(low=0, high=7, shape=(self.ROWS, self.COLS), dtype=np.uint8)
        
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake Gym Environment with Penalty")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake_positions = self.generate_snake()
        self.fruit_position = self.generate_fruit(self.snake_positions)
        self.direction = 'RIGHT'
        self.score = 0
        self.steps_without_fruit = 0  # Counter for steps without eating fruit
        
        return self._get_observation()

    def step(self, action):
        # Translate action to direction
        if action == 0 and self.direction != 'DOWN':
            self.direction = 'UP'
        elif action == 1 and self.direction != 'UP':
            self.direction = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif action == 3 and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        self.snake_positions, self.fruit_position, points, game_over, self_collision = self.move_snake(self.snake_positions, self.direction, self.fruit_position)
        self.score += points

        reward = points - 0.01  # Add a smaller penalty for each step
        if self_collision:
            reward = -5  # Directly set the reward to -5 for self-collision
        done = game_over

        observation = self._get_observation()

        # Optional: Additional penalty for too many steps without eating fruit
        self.steps_without_fruit += 1
        if self.steps_without_fruit > 100:
            reward -= 1
            done = True

        return observation, reward, done, {}

    def render(self, mode='human', close=False):
        self.window.fill((255, 255, 255))
        self.draw_grid()
        self.draw_snake(self.snake_positions)
        self.draw_fruit(self.fruit_position)
        
        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(15)  # Increase the frame rate for smoother gameplay

    def close(self):
        pygame.quit()

    def draw_grid(self):
        for row in range(self.ROWS):
            for col in range(self.COLS):
                rect = pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.window, BROWN, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # Draw cell borders

    def draw_snake(self, snake_positions):
        for i, pos in enumerate(snake_positions):
            rect = pygame.Rect(pos[1] * self.CELL_SIZE, pos[0] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.window, SNAKE_COLOR, rect)
            if i == 0:  # Draw the head of the snake
                center = (pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2, pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2)
                pygame.draw.circle(self.window, HEAD_COLOR, center, self.CELL_SIZE // 4)

    def draw_fruit(self, fruit_position):
        rect = pygame.Rect(fruit_position[1] * self.CELL_SIZE, fruit_position[0] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.window, FRUIT_COLOR, rect)

    def generate_snake(self):
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            row = random.randint(0, self.ROWS - 1)
            col_start = random.randint(0, self.COLS - 2)
            snake_positions = [(row, col_start), (row, col_start + 1)]
        else:
            col = random.randint(0, self.COLS - 1)
            row_start = random.randint(0, self.ROWS - 2)
            snake_positions = [(row_start, col), (row_start + 1, col)]
        return snake_positions

    def generate_fruit(self, snake_positions):
        while True:
            fruit_position = (random.randint(0, self.ROWS - 1), random.randint(0, self.COLS - 1))
            if fruit_position not in snake_positions:
                return fruit_position

    def move_snake(self, snake_positions, direction, fruit_position):
        head_x, head_y = snake_positions[0]
        if direction == 'UP':
            new_head = (head_x - 1, head_y)
        elif direction == 'DOWN':
            new_head = (head_x + 1, head_y)
        elif direction == 'LEFT':
            new_head = (head_x, head_y - 1)
        elif direction == 'RIGHT':
            new_head = (head_x, head_y + 1)

        self_collision = False
        # Check if the snake's head goes out of bounds
        if not (0 <= new_head[0] < self.ROWS and 0 <= new_head[1] < self.COLS):
            return snake_positions, fruit_position, -5, True, self_collision  # Game over with -5 reward for boundary collision

        # Check if the snake's head overlaps with any part of its body
        if new_head in snake_positions:
            self_collision = True
            return snake_positions, fruit_position, -5, True, self_collision  # Game over with -5 reward for self-collision

        # Check if the snake's head overlaps with the fruit
        if new_head == fruit_position:
            fruit_position = self.generate_fruit(snake_positions)  # Generate new fruit position
            self.steps_without_fruit = 0  # Reset counter
            return [new_head] + snake_positions, fruit_position, 10, False, self_collision  # Return 10 points and grow the snake
        else:
            # Add new head and remove tail, ensuring the snake grows
            return [new_head] + snake_positions[:-1], fruit_position, 0, False, self_collision  # No points for scoring

    def _get_observation(self):
        # Create a blank grid
        grid = np.full((self.ROWS, self.COLS), 0, dtype=np.uint8)
        
        # Set snake position in the grid
        for pos in self.snake_positions[1:]:
            grid[pos[0], pos[1]] = 1  # 1 for snake body
        head_pos = self.snake_positions[0]
        grid[head_pos[0], head_pos[1]] = 7  # 7 for snake head
        
        # Set fruit position in the grid
        grid[self.fruit_position[0], self.fruit_position[1]] = 2  # 2 for fruit

        return grid



# # To use this environment
if __name__ == "__main__":
    for i in range(20):
        env = SnakeEnvWithPenalty(width=600, height=600, rows=10, cols=10)

        observation = env.reset()
        rewards = 0

        for step in range(100):
            
            env.render()
            print(f"Observation: \n {observation}")
            action = env.action_space.sample()  # Take a random action
            print(f"Action: {action}")
            observation, reward, done, info = env.step(action)
            print(f"Observation: \n {observation}")
            print("*****************************")
            rewards = rewards + reward
            print(f"{step}:{reward}: {rewards}")
            if done:
                observation = env.reset()
                print(rewards)
                break
