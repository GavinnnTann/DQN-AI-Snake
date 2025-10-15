"""
Constants module for the Snake Game.
Contains game settings, colors, and other constants.
"""

import os

# Get the directory of this file (advanced_snake folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GAME_WIDTH = 600
GAME_HEIGHT = 600
INFO_WIDTH = 200
GRID_SIZE = 20
GRID_WIDTH = GAME_WIDTH // GRID_SIZE
GRID_HEIGHT = GAME_HEIGHT // GRID_SIZE

# Training window dimensions
TRAINING_SCREEN_WIDTH = 1400   # ENHANCED: Increased from 1300 for better layout
TRAINING_SCREEN_HEIGHT = 900   # ENHANCED: Increased from 800 for button visibility
MIN_TRAINING_SCREEN_WIDTH = 1200  # ENHANCED: Increased from 800
MIN_TRAINING_SCREEN_HEIGHT = 700   # ENHANCED: Reasonable minimum (was 600)

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 155, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)

# Game settings
INITIAL_SNAKE_LENGTH = 3
FRAME_RATES = {"Slow": 5, "Medium": 10, "Fast": 15, "Very Fast": 100}
DEFAULT_SPEED = "Very Fast"
GAME_TITLE = "Advanced Snake Game"

# Game modes
MANUAL_MODE = "Manual"
ASTAR_MODE = "A* Algorithm"
DIJKSTRA_MODE = "Dijkstra Algorithm"
QLEARNING_MODE = "Q-Learning Algorithm"
GAME_MODES = [MANUAL_MODE, ASTAR_MODE, DIJKSTRA_MODE, QLEARNING_MODE]

# Direction vectors (row, col)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

# Key mappings
KEY_UP = "w"
KEY_DOWN = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"

# Game states
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"
STATE_PAUSED = "paused"

# Score settings
POINTS_PER_FOOD = 10

# Q-learning parameters
QLEARNING_ALPHA = 0.1       # Learning rate
QLEARNING_GAMMA = 0.9       # Discount factor
QLEARNING_EPSILON = 1.0     # Initial exploration rate
QLEARNING_EPSILON_MIN = 0.01  # Minimum exploration rate
QLEARNING_EPSILON_DECAY = 0.995  # Decay rate for exploration
QLEARNING_BATCH_SIZE = 64   # Batch size for training

# Training settings
DEFAULT_TRAINING_EPISODES = 1000
TRAINING_EPISODES_OPTIONS = [100, 500, 1000, 5000, 10000]
MODEL_SAVE_INTERVAL = 100   # Save model every N episodes
TRAINING_DISPLAY_INTERVAL = 1  # Update display every N episodes

# Q-learning rewards
REWARD_FOOD = 10.0         # Reward for eating food
REWARD_DEATH = -12.0       # INCREASED penalty - make death more costly
REWARD_MOVE_TOWARDS_FOOD = 0.15  # INCREASED from 0.1 - stronger signal for good moves
REWARD_MOVE_AWAY_FROM_FOOD = -0.15  # INCREASED penalty - discourage bad moves more
REWARD_SURVIVAL = 0.02     # INCREASED from 0.01 - reward staying alive more

# Model file paths
QMODEL_DIR = os.path.join(SCRIPT_DIR, "models")
QMODEL_FILE = "snake_qlearning_model.pkl"
DQN_MODEL_FILE = "snake_dqn_model.pth"

# DQN parameters
DQN_MODE = "Advanced DQN"  # Mode name for the menu
DQN_LEARNING_RATE = 0.003   # PERFORMANCE BOOST: Increased from 0.001 for faster learning (3x faster weight updates)
DQN_GAMMA = 0.96           # REDUCED from 0.99 - focus more on immediate rewards
DQN_EPSILON = 1.0          # Initial exploration rate
DQN_EPSILON_MIN = 0.01     # Minimum exploration rate
DQN_EPSILON_DECAY = 0.95  # Decay rate for exploration (per episode, not per step)
DQN_BATCH_SIZE = 64        # Batch size for training (can be increased for GPU)
DQN_MEMORY_SIZE = 100000   # Size of replay buffer
DQN_TARGET_UPDATE = 20     # REDUCED from 25 - update target more frequently for faster adaptation
DQN_PRIORITIZED_ALPHA = 0.6  # Alpha parameter for prioritized replay (0 = uniform sampling)
DQN_PRIORITIZED_BETA = 0.4   # Beta parameter for prioritized replay (importance sampling)
DQN_BETA_INCREMENT = 0.001   # How much to increase beta each sampling

# CUDA/GPU settings
USE_CUDA = True            # Whether to use CUDA when available
GPU_BATCH_SIZE = 512       # SPEED OPTIMIZATION: Increased from 256 for smoother gradients (faster convergence)
CPU_BATCH_SIZE = 128       # PERFORMANCE BOOST: Increased from 64 for better convergence

# Neural network architecture
DQN_HIDDEN_SIZE = 128      # Size of hidden layers
DQN_LEARNING_STARTS = 2000  # SPEED OPTIMIZATION: Increased from 1000 for more diverse initial experiences

# Training settings for DQN
DQN_TRAINING_EPISODES_OPTIONS = [100, 500, 1000, 5000, 10000]
DEFAULT_DQN_TRAINING_EPISODES = 1000
DQN_MODEL_SAVE_INTERVAL = 100  # Save model every N episodes

# ============================================================
# ENHANCED DQN DECAY PARAMETERS
# ============================================================
# These parameters control epsilon and learning rate decay
# for the Enhanced DQN agent during curriculum learning.
# Edit these values to adjust exploration/exploitation balance.

# Epsilon Decay Rates (per episode)
# Lower value = faster decay (less exploration sooner)
# Higher value = slower decay (more exploration longer)
# Formula: epsilon_new = epsilon_old × decay_rate
# Half-life examples: 0.9965→198 eps, 0.995→138 eps, 0.990→69 eps
STAGE_EPSILON_DECAY = {
    0: 0.98,    # Stage 0: Slower decay (half-life ~200 episodes)
    1: 0.985,   # Stage 1: Medium decay (half-life ~138 episodes)
    2: 0.995,   # Stage 2: SLOWER decay - maintain exploration longer
    3: 0.997,   # Stage 3: Very slow decay - keep exploring
    4: 0.998    # Stage 4: Minimal decay - continue learning
}

# Epsilon Minimum Values (per stage)
# Higher value = more exploration even late in training
# Lower value = more exploitation late in training
STAGE_EPSILON_MINIMUMS = {
    0: 0.1,   # Stage 0: Min 10% random actions
    1: 0.05,  # Stage 1: Min 5% random actions
    2: 0.05,  # Stage 2: Min 5% random actions - KEEP EXPLORING
    3: 0.04,  # Stage 3: Min 4% random actions - still exploring
    4: 0.03   # Stage 4: Min 3% random actions - increased from 1%
}

# Learning Rate Decay Rates (per episode)
# Lower value = faster decay (quicker stabilization)
# Higher value = slower decay (maintains plasticity longer)
# Formula: learning_rate_new = learning_rate_old × decay_rate
STAGE_LR_DECAY = {
    0: 0.9990,  # Stage 0: Slower decay to maintain strong learning
    1: 0.9993,  # Stage 1: Slower decay
    2: 0.9997,  # Stage 2: MUCH slower - keep learning strong
    3: 0.9998,  # Stage 3: Minimal decay - maintain plasticity
    4: 0.9999   # Stage 4: Almost no decay - keep adapting
}

# Learning Rate Minimum Values (per stage)
# Higher value = keeps learning stronger
# Lower value = more fine-tuning capability
STAGE_LR_MINIMUMS = {
    0: 0.002,   # Stage 0: Decay from 0.005 down to 0.002
    1: 0.0015,  # Stage 1: Decay from 0.003 down to 0.0015
    2: 0.0012,  # Stage 2: HIGHER minimum - keep learning strong
    3: 0.0008,  # Stage 3: HIGHER minimum - maintain plasticity
    4: 0.0005   # Stage 4: HIGHER minimum - keep adapting
}

# ============================================================
# STUCK DETECTION PARAMETERS
# ============================================================
# Controls when and how the agent gets an epsilon boost to escape
# local optima. Based on analysis, boosts may be too aggressive.

# Enable/Disable stuck detection entirely
ENABLE_STUCK_DETECTION = True  # Set to False to disable all boosts

# How many consecutive stuck checks before triggering boost
# Higher value = more conservative (less frequent boosts)
# Lower value = more aggressive (more frequent boosts)
# Default: 3 means 3 × 50 episodes = 150 episodes of being stuck
STUCK_COUNTER_THRESHOLD = 3  # Range: 1-10 (1=very aggressive, 10=very conservative)

# Cooldown period between boosts (in episodes)
# Prevents oscillation from too-frequent boosts
# Higher value = longer wait between boosts
STUCK_BOOST_COOLDOWN = 200  # Range: 50-500 episodes

# Epsilon boost amount when stuck is detected
# How much to increase epsilon when agent is stuck
# Higher value = more exploration after boost
# Lower value = gentler boost
STUCK_EPSILON_BOOST = 0.10  # Range: 0.05-0.30 (0.10 = 10% increase)

# Maximum epsilon after boost
# Caps how high epsilon can go from boosts
STUCK_EPSILON_MAX = 0.4  # Range: 0.3-0.6

# Improvement threshold for stuck detection
# How much average score must improve to NOT be considered stuck
# Higher value = harder to avoid being marked as stuck
# Lower value = easier to avoid being marked as stuck
STUCK_IMPROVEMENT_THRESHOLD = 5.0  # Range: 2.0-15.0 points

# Variance threshold for stuck detection
# Maximum score variance to be considered stuck
# Lower value = requires more consistent scores to be stuck
# Higher value = allows more variance while stuck
STUCK_VARIANCE_THRESHOLD = 100.0  # Range: 50.0-500.0