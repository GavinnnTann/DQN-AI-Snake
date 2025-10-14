"""
Game engine module for the Snake Game.
Handles the core game mechanics like snake movement, food generation, and collision detection.
"""

import random
import pygame
import torch
from collections import deque
from constants import *
from gpu_utils import device

class GameEngine:
    def __init__(self):
        """Initialize the game engine."""
        self.reset_game()
    
    def reset_game(self):
        """Reset the game state."""
        # Initialize snake in the middle of the grid
        self.snake = deque()
        start_row, start_col = GRID_HEIGHT // 2, GRID_WIDTH // 2
        
        # Create initial snake body (3 segments)
        for i in range(INITIAL_SNAKE_LENGTH):
            self.snake.append((start_row, start_col - i))
        
        # Set initial direction (right)
        self.direction = RIGHT
        self.next_direction = self.direction  # Store next direction to prevent 180-degree turns
        
        # Game state variables
        self.score = 0
        self.game_over = False
        self.paused = False
        
        # Generate initial food
        self.food = self.generate_food()
    
    def generate_food(self):
        """Generate food in a random position not occupied by the snake."""
        while True:
            food_pos = (random.randint(0, GRID_HEIGHT - 1), 
                        random.randint(0, GRID_WIDTH - 1))
            # Make sure food doesn't appear on the snake
            if food_pos not in self.snake:
                return food_pos
    
    def change_direction(self, new_direction):
        """Change the snake's direction, preventing 180-degree turns."""
        # Prevent 180-degree turns (e.g., from UP to DOWN)
        opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if new_direction != opposite_directions.get(self.direction):
            self.next_direction = new_direction
    
    def move_snake(self):
        """Move the snake in its current direction."""
        if self.game_over or self.paused:
            return
        
        # Update the current direction to the next direction
        self.direction = self.next_direction
        
        # Get the current head position
        head_row, head_col = self.snake[0]
        
        # Calculate new head position based on direction
        new_row = head_row + self.direction[0]
        new_col = head_col + self.direction[1]
        
        # Check for boundary collision (game over condition)
        if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
            self.game_over = True
            return
            
        new_head = (new_row, new_col)
        
        # Check for collision with self (game over condition)
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head to the snake
        self.snake.appendleft(new_head)
        
        # Check if snake ate food
        if new_head == self.food:
            # Snake grows, generate new food, increase score
            self.score += POINTS_PER_FOOD
            self.food = self.generate_food()
        else:
            # Remove the tail if no food was eaten
            self.snake.pop()
    
    def get_game_state(self):
        """Return the current game state for AI algorithms."""
        state = {
            'snake': list(self.snake),
            'food': self.food,
            'grid_width': GRID_WIDTH,
            'grid_height': GRID_HEIGHT,
            'score': self.score,
            'game_over': self.game_over
        }
        return state
    
    def get_snake_head(self):
        """Return the position of the snake's head."""
        return self.snake[0]
    
    def get_snake_body(self):
        """Return the positions of the snake's body segments (excluding head)."""
        return list(self.snake)[1:]
    
    def is_valid_position(self, position):
        """Check if a position is within the game grid."""
        row, col = position
        return 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH
    
    def is_position_safe(self, position):
        """Check if a position is safe (not occupied by snake)."""
        return position not in self.snake
    
    def get_valid_moves(self):
        """Get list of valid moves from current position (for algorithms)."""
        head_row, head_col = self.get_snake_head()
        possible_moves = [UP, DOWN, LEFT, RIGHT]
        valid_moves = []
        
        for move in possible_moves:
            new_row = head_row + move[0]
            new_col = head_col + move[1]
            
            # Check if the new position is within boundaries
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue  # Skip this move as it leads outside the boundary
                
            new_pos = (new_row, new_col)
            
            # A move is valid if it doesn't result in collision with snake body
            # We exclude the tail check if the snake is about to grow
            snake_body = self.get_snake_body()
            if len(snake_body) > 0 and self.get_snake_head() != self.food:
                # The tail will move, so we can move into the tail's current position
                snake_body = snake_body[:-1]
                
            if new_pos not in snake_body:
                valid_moves.append(move)
                
        return valid_moves
    
    def set_direction_from_algorithm(self, direction):
        """Set the direction based on algorithm output with safety check."""
        # Explicitly prevent 180° reversals that would cause immediate game over
        opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        current_direction = self.direction
        
        if direction == opposite_directions.get(current_direction):
            # If algorithm tries to make illegal 180° turn, maintain current direction
            return
            
        self.change_direction(direction)
        
    def get_distance(self, pos1, pos2):
        """Calculate the Manhattan distance between two positions."""
        row1, col1 = pos1
        row2, col2 = pos2
        return abs(row1 - row2) + abs(col1 - col2)
    
    def get_state(self):
        """
        Convert the current game state to a tensor representation for the neural network.
        This method is used by the DQN agent for state representation.
        """
        snake = self.snake
        head = snake[0]
        food = self.food
        
        # Detect danger in each direction
        # 1 = danger, 0 = safe
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Get the current direction
        current_direction = self.direction
        
        # Check danger straight
        next_pos_straight = (head[0] + current_direction[0], head[1] + current_direction[1])
        if (next_pos_straight[0] < 0 or next_pos_straight[0] >= GRID_HEIGHT or
            next_pos_straight[1] < 0 or next_pos_straight[1] >= GRID_WIDTH or
            next_pos_straight in list(snake)[:-1]):  # Skip checking collision with tail
            danger_straight = 1
        
        # Get right and left directions relative to current direction
        if current_direction == UP:
            right_dir = RIGHT
            left_dir = LEFT
        elif current_direction == RIGHT:
            right_dir = DOWN
            left_dir = UP
        elif current_direction == DOWN:
            right_dir = LEFT
            left_dir = RIGHT
        else:  # LEFT
            right_dir = UP
            left_dir = DOWN
        
        # Check danger right
        next_pos_right = (head[0] + right_dir[0], head[1] + right_dir[1])
        if (next_pos_right[0] < 0 or next_pos_right[0] >= GRID_HEIGHT or
            next_pos_right[1] < 0 or next_pos_right[1] >= GRID_WIDTH or
            next_pos_right in list(snake)[:-1]):
            danger_right = 1
        
        # Check danger left
        next_pos_left = (head[0] + left_dir[0], head[1] + left_dir[1])
        if (next_pos_left[0] < 0 or next_pos_left[0] >= GRID_HEIGHT or
            next_pos_left[1] < 0 or next_pos_left[1] >= GRID_WIDTH or
            next_pos_left in list(snake)[:-1]):
            danger_left = 1
        
        # Food direction relative to snake head
        food_left = int(food[1] < head[1])
        food_right = int(food[1] > head[1])
        food_up = int(food[0] < head[0])
        food_down = int(food[0] > head[0])
        
        # Current snake direction
        dir_up = int(current_direction == UP)
        dir_right = int(current_direction == RIGHT)
        dir_down = int(current_direction == DOWN)
        dir_left = int(current_direction == LEFT)
        
        # Create state representation as a tensor - shape [11]
        state = torch.tensor([
            danger_straight,
            danger_right,
            danger_left,
            food_up,
            food_right,
            food_down,
            food_left,
            dir_up,
            dir_right,
            dir_down,
            dir_left
        ], dtype=torch.float32, device=device)
        
        return state
    
    def render(self, screen):
        """Render the game state to the screen."""
        # Calculate offset to center the game area
        from constants import SCREEN_WIDTH, SCREEN_HEIGHT
        offset_x = (SCREEN_WIDTH - GAME_WIDTH) // 2
        offset_y = (SCREEN_HEIGHT - GAME_HEIGHT) // 2
        
        # Clear the entire screen first
        screen.fill(BLACK)
        
        # Clear the game area
        pygame.draw.rect(screen, BLACK, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        
        # Draw grid lines
        for i in range(0, GRID_WIDTH + 1):
            pygame.draw.line(screen, DARK_GRAY, (offset_x + i * GRID_SIZE, offset_y), 
                            (offset_x + i * GRID_SIZE, offset_y + GAME_HEIGHT), 1)
        for i in range(0, GRID_HEIGHT + 1):
            pygame.draw.line(screen, DARK_GRAY, (offset_x, offset_y + i * GRID_SIZE), 
                            (offset_x + GAME_WIDTH, offset_y + i * GRID_SIZE), 1)
        
        # Draw boundaries
        boundary_thickness = 3
        pygame.draw.rect(screen, BLUE, (offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT), boundary_thickness)
        
        # Draw food
        food_row, food_col = self.food
        food_rect = pygame.Rect(
            offset_x + food_col * GRID_SIZE, 
            offset_y + food_row * GRID_SIZE, 
            GRID_SIZE, 
            GRID_SIZE
        )
        pygame.draw.rect(screen, RED, food_rect)
        
        # Draw snake
        for i, (row, col) in enumerate(self.snake):
            snake_segment = pygame.Rect(
                offset_x + col * GRID_SIZE, 
                offset_y + row * GRID_SIZE, 
                GRID_SIZE, 
                GRID_SIZE
            )
            
            if i == 0:  # Head
                pygame.draw.rect(screen, DARK_GREEN, snake_segment)
                # Draw eyes
                eye_size = GRID_SIZE // 5
                eye_offset = GRID_SIZE // 4
                
                # Left eye
                left_eye_x = offset_x + col * GRID_SIZE + eye_offset
                left_eye_y = offset_y + row * GRID_SIZE + eye_offset
                pygame.draw.circle(screen, WHITE, (left_eye_x, left_eye_y), eye_size)
                
                # Right eye
                right_eye_x = offset_x + col * GRID_SIZE + GRID_SIZE - eye_offset
                right_eye_y = offset_y + row * GRID_SIZE + eye_offset
                pygame.draw.circle(screen, WHITE, (right_eye_x, right_eye_y), eye_size)
            else:  # Body
                pygame.draw.rect(screen, GREEN, snake_segment)