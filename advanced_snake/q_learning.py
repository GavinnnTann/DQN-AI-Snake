"""
Q-Learning module for the Snake Game.
Implements a Q-learning agent to learn optimal snake movement strategy.
"""

import numpy as np
import random
import os
import pickle
from collections import deque
from constants import *

class SnakeQLearningAgent:
    def __init__(self, game_engine):
        """Initialize the Q-learning agent with a reference to the game engine."""
        self.game_engine = game_engine
        self.q_table = {}  # State-action value function
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.epsilon = QLEARNING_EPSILON  # Exploration rate
        self.alpha = QLEARNING_ALPHA  # Learning rate
        self.gamma = QLEARNING_GAMMA  # Discount factor
        
        # Statistics for training visualization
        self.stats = {
            'scores': [],
            'steps': [],
            'epsilon': [],
            'q_values': [],
            'rewards': []
        }
    
    def get_state(self):
        """
        Convert the current game state to a simplified representation for Q-learning.
        Returns a tuple representation of the state.
        """
        snake = self.game_engine.snake
        head = snake[0]
        food = self.game_engine.food
        
        # Detect danger in each direction
        # 1 = danger, 0 = safe
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Get the current direction
        current_direction = self.game_engine.direction
        
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
        food_left = food[1] < head[1]
        food_right = food[1] > head[1]
        food_up = food[0] < head[0]
        food_down = food[0] > head[0]
        
        # Current snake direction
        dir_up = current_direction == UP
        dir_right = current_direction == RIGHT
        dir_down = current_direction == DOWN
        dir_left = current_direction == LEFT
        
        # Return a simplified state representation
        return (
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
        )
    
    def get_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        Returns the selected action (UP, DOWN, LEFT, RIGHT).
        """
        # Exploration: choose a random action with probability epsilon
        if random.random() < self.epsilon:
            valid_moves = self.game_engine.get_valid_moves()
            if valid_moves:
                return random.choice(valid_moves)
            return random.choice([UP, DOWN, LEFT, RIGHT])
        
        # Exploitation: choose the action with the highest Q-value
        return self._get_best_action(state)
    
    def _get_best_action(self, state):
        """
        Get the action with the highest Q-value for the given state.
        """
        # If state is not in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = {
                UP: 0.0,
                DOWN: 0.0,
                LEFT: 0.0,
                RIGHT: 0.0
            }
        
        # Filter actions by valid moves if possible
        valid_moves = self.game_engine.get_valid_moves()
        if valid_moves:
            # Get Q-values for valid actions
            valid_q_values = {move: self.q_table[state].get(move, 0.0) for move in valid_moves}
            # Return action with highest Q-value
            return max(valid_q_values, key=valid_q_values.get)
        
        # If no valid moves (rare case), return action with highest overall Q-value
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=QLEARNING_BATCH_SIZE):
        """
        Train the agent using experience replay.
        Randomly samples a batch from memory and updates Q-values.
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch from memory
        batch = random.sample(self.memory, batch_size)
        total_q_change = 0
        
        for state, action, reward, next_state, done in batch:
            # Initialize state in Q-table if needed
            if state not in self.q_table:
                self.q_table[state] = {
                    UP: 0.0,
                    DOWN: 0.0,
                    LEFT: 0.0,
                    RIGHT: 0.0
                }
            
            # Calculate target Q-value
            if done:
                target = reward
            else:
                # Get max Q-value for next state
                if next_state not in self.q_table:
                    self.q_table[next_state] = {
                        UP: 0.0,
                        DOWN: 0.0,
                        LEFT: 0.0,
                        RIGHT: 0.0
                    }
                max_next_q = max(self.q_table[next_state].values())
                target = reward + self.gamma * max_next_q
            
            # Record old Q-value for tracking learning progress
            old_q = self.q_table[state].get(action, 0.0)
            
            # Update Q-value using Q-learning update rule
            self.q_table[state][action] = old_q + self.alpha * (target - old_q)
            
            # Track change in Q-values for monitoring learning
            total_q_change += abs(self.q_table[state][action] - old_q)
        
        # Track average Q-value change for this batch
        avg_q_change = total_q_change / batch_size
        self.stats['q_values'].append(avg_q_change)
        
        # Decay exploration rate
        if self.epsilon > QLEARNING_EPSILON_MIN:
            self.epsilon *= QLEARNING_EPSILON_DECAY
            self.stats['epsilon'].append(self.epsilon)
    
    def calculate_reward(self, old_state, new_state, action_taken, game_over, ate_food):
        """
        Calculate the reward for a given state transition.
        Rewards eating food and surviving, penalizes death.
        """
        reward = 0
        
        # Major rewards/penalties
        if game_over:
            reward += REWARD_DEATH
        elif ate_food:
            reward += REWARD_FOOD
        else:
            reward += REWARD_SURVIVAL  # Small reward for surviving
        
        # Check if moved closer to food
        head = self.game_engine.get_snake_head()
        food = self.game_engine.food
        old_head = self.game_engine.snake[0]
        
        old_distance = abs(old_head[0] - food[0]) + abs(old_head[1] - food[1])
        new_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
        
        if new_distance < old_distance:
            reward += REWARD_MOVE_TOWARDS_FOOD
        elif new_distance > old_distance:
            reward += REWARD_MOVE_AWAY_FROM_FOOD
        
        return reward
    
    def train_step(self):
        """
        Execute one training step:
        1. Get current state
        2. Choose action
        3. Take action and observe reward and next state
        4. Remember experience
        5. Update Q-values
        Returns (reward, done, ate_food)
        """
        # Get current state
        current_state = self.get_state()
        
        # Choose action
        action = self.get_action(current_state)
        
        # Store old state info for reward calculation
        old_score = self.game_engine.score
        
        # Take action
        self.game_engine.set_direction_from_algorithm(action)
        self.game_engine.move_snake()
        
        # Observe new state
        new_state = self.get_state()
        done = self.game_engine.game_over
        ate_food = self.game_engine.score > old_score
        
        # Calculate reward
        reward = self.calculate_reward(current_state, new_state, action, done, ate_food)
        self.stats['rewards'].append(reward)
        
        # Store experience in memory
        self.remember(current_state, action, reward, new_state, done)
        
        # Train using experience replay
        self.replay()
        
        return reward, done, ate_food
    
    def save_model(self, filepath):
        """Save the Q-table to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model (Q-table) to a pickle file
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table, 
                'stats': self.stats,
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, filepath):
        """Load the Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.stats = data.get('stats', self.stats)
                self.epsilon = data.get('epsilon', QLEARNING_EPSILON_MIN)
            return True
        return False
    
    def get_next_move_qlearning(self):
        """Get the next move based on trained Q-learning policy (for gameplay)."""
        current_state = self.get_state()
        return self._get_best_action(current_state)