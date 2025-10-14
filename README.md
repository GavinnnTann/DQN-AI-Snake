# DQN-AI-Snake

A Snake game implementation with a Deep Q-Network (DQN) AI agent that learns to play the game through reinforcement learning.

## Features

- Classic Snake game implementation using Pygame
- Deep Q-Network (DQN) reinforcement learning agent
- Experience replay buffer for stable training
- Epsilon-greedy exploration strategy
- Real-time visualization of gameplay
- Model saving and loading capabilities

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GavinnnTann/DQN-AI-Snake.git
cd DQN-AI-Snake
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the AI Agent

To train the DQN agent:

```bash
python agent.py
```

The agent will:
- Start playing the game with random moves
- Gradually learn optimal strategies through trial and error
- Save the best model when it achieves a new high score
- Print progress information after each game

Training progress shows:
- Current game number
- Score achieved in that game
- Record high score

The trained model will be saved in the `model/` directory as `model.pth`.

### How It Works

#### Game Environment (`snake_game.py`)
- Implements the Snake game with Pygame
- Provides an interface for the AI agent to interact with
- Returns state, reward, and game over status after each action

#### DQN Model (`model.py`)
- Neural network with:
  - Input layer: 11 neurons (game state features)
  - Hidden layer: 256 neurons with ReLU activation
  - Output layer: 3 neurons (actions: straight, right, left)
- Uses MSE loss and Adam optimizer
- Implements Q-learning update rule

#### AI Agent (`agent.py`)
- State representation (11 features):
  - Danger detection (straight, right, left)
  - Current direction (4 binary values)
  - Food location relative to head (4 binary values)
- Action space: [straight, turn right, turn left]
- Epsilon-greedy exploration (decreases over time)
- Experience replay with batch training
- Reward system:
  - +10 for eating food
  - -10 for collision/game over
  - 0 for regular moves

## Project Structure

```
DQN-AI-Snake/
├── agent.py          # DQN agent implementation and training loop
├── model.py          # Neural network model and trainer
├── snake_game.py     # Snake game environment
├── helper.py         # Plotting utilities
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Training Tips

- The agent starts with random exploration (high epsilon)
- Epsilon decreases as more games are played
- Training typically shows improvement after 50-100 games
- The model automatically saves when a new record is achieved
- You can stop training anytime (Ctrl+C) and resume later

## License

MIT License - see LICENSE file for details