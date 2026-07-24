# Advanced Snake Game with Deep Reinforcement Learning

An advanced implementation of the classic Snake game featuring multiple AI agents, a Deep Q-Network, and a safety layer that lets the trained snake **win the game** — filling all 900 cells (score 8960) on the 30×30 board.

The Enhanced DQN is wrapped by a **survival shield** and a **Hamiltonian cycle backbone** (`safety.py`) that guarantees the snake never traps itself and can complete a perfect game. See [Winning the Game](#-winning-the-game-survival-shield--cycle-backbone).

---

## Quick Start Guide

### Just Want to Play?

```bash
# 1. Install dependencies
pip install torch pygame numpy matplotlib

# 2. Play the game
python play_snake.py
```

**Controls:** Use **WASD** or **Arrow Keys** to move, press **G** for debug mode, **P** to pause.

---

### Want to Train an AI?

```bash
# 1. Install dependencies (if not done)
pip install torch pygame numpy matplotlib tkinter psutil pandas

# 2. Launch training interface
python train_snake.py
```

**Quick Training Steps:**
1. Select "Enhanced DQN" (recommended)
2. Set episodes to **1000** or more
3. Click **"Start Training"**
4. Watch the AI learn in real-time!
5. After training, play the game and select **"Enhanced DQN"** mode

**Typical Results:**
- **Playing with the cycle backbone (default): a perfect game every time** — the snake fills the board (score 8960), even with an untrained model. The DQN's training only makes it *faster*, not *win-or-not*.
- Playing in pure survival-shield mode (backbone off): plays far past the old ~1000-point ceiling and never traps itself; score depends on how well the model is trained.
- Training (shield mode) sharpens the network's food-seeking; watch the running-average score climb in the training UI.

---

### 5-Minute Setup

**Complete setup from scratch:**

```bash
# 1. Clone/download this repository
# 2. Navigate to Snake Game folder
cd "Snake Game"

# 3. Install requirements
pip install -r requirements.txt

# 4. Try manual play first
python play_snake.py
# Select "Manual" mode and play!

# 5. Train your first AI
python train_snake.py
# Click "Start Training", wait 10-15 minutes

# 6. Watch your AI play
python play_snake.py
# Select "Enhanced DQN" mode
```

**That's it!** You now have a trained AI snake player.

---

## Table of Contents

-  [Quick Start Guide](#-quick-start-guide)
- [Winning the Game: Survival Shield & Cycle Backbone](#-winning-the-game-survival-shield--cycle-backbone)
- [Features](#-features)
  - [Game Modes](#game-modes)
  - [Training System](#training-system)
  - [Training UI Features](#training-ui-features)
  - [Debug Mode](#debug-mode)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Playing the Game](#playing-the-game)
  - [Training AI Models](#training-ai-models)
- [Training Tips](#-training-tips)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Known Issues](#-known-issues)
- [Version History](#-version-history)

---

## Features

### Game Modes

1. **Manual Mode**: Full player control using WASD or arrow keys
2. **A* Algorithm**: Watch optimal pathfinding in action
3. **Dijkstra Algorithm**: Classic shortest-path algorithm demonstration
4. **Hamiltonian Cycle**: Guaranteed win strategy - visits every cell exactly once (see [HAMILTONIAN_ALGORITHM.md](HAMILTONIAN_ALGORITHM.md))
5. **Q-Learning AI**: Traditional tabular reinforcement learning
6. **DQN AI**: Deep Q-Network with 11-feature state representation
7. **Enhanced DQN**: Advanced 34-feature DQN with:
   - A* reward shaping for guidance
   - Trap detection and extended danger sensing
   - Body proximity awareness
   - **Survival shield + Hamiltonian cycle backbone** so the trained snake can win the game (see below)

### 🏆 Winning the Game: Survival Shield & Cycle Backbone

A learned policy alone traps itself once the snake gets long (the classic Snake ceiling). The Enhanced DQN is wrapped by a safety layer in [`advanced_snake/safety.py`](advanced_snake/safety.py) with **two modes**, toggled by `agent.use_cycle_backbone`:

**🏆 Cycle Backbone (default in play — guaranteed win)**
- The snake follows a **Hamiltonian cycle** (a fixed route that visits every cell exactly once), so it can always fill the board without ever colliding.
- The network is still allowed to take **shortcuts** — but only ones that provably (1) don't overtake the tail in cycle order and (2) advance toward the food without skipping past it. Every allowed shortcut preserves the win, so the network can only make the snake *faster*, never kill it.
- On a fresh game the snake is aligned to the cycle (it "snaps" to the start corner — this is expected).
- **Result: a perfect game (score 8960, all 899/900 cells) every time**, verified on boards from 8×8 to 30×30, even with an untrained network. A full 30×30 win is ~14.5s of AI compute.

**🛡️ Survival Shield (`use_cycle_backbone = False`)**
- The network drives freely; the shield simulates each candidate move and **vetoes** any that would hit a wall/body or trap the snake, using **tail-reachability** (can the head still reach the tail?) and flood-fill free-space checks.
- A **loop-breaker** forces progress toward food (shortest safe path) if the snake dawdles, so it never gets stuck circling.
- High score and high variance, but **not** guaranteed to fill the board — this is the "pure trained snake" showcase.

> The game's older `Hamiltonian Cycle` and `DHCR` menu modes are separate hand-coded algorithms and are independent of the DQN safety layer above.

### Training System

#### **Enhanced DQN (Recommended)**
- **34 Features**: Comprehensive state representation
  - Danger detection (13 features): immediate, extended, traps, body proximity, wall distances
  - Food information (6 features): direction and distances
  - Navigation (12 features): current direction, available space, snake length, tail tracking
  - A* hints (3 features): suggested actions from A* pathfinding

- **Curriculum Learning**: Progressive difficulty stages
  - Stage 0 (0-25 score): High A* guidance (0.5), high exploration (0.20)
  - Stage 1 (25-60): Reduced guidance (0.35), moderate exploration (0.15)
  - Stage 2 (60-120): Lower guidance (0.20), balanced exploration (0.10)
  - Stage 3 (120-250): Minimal guidance (0.10), focused exploration (0.05)
  - Stage 4 (250+): Independent learning, minimal exploration (0.01)

- **Advanced Techniques**:
  - **Double DQN**: online net selects the next action, target net evaluates it — removes the max-operator overestimation bias
  - **Dueling Network**: separate value and advantage streams
  - **Huber (Smooth L1) loss**: robust to the occasional large reward (eating/dying) that MSE would over-weight
  - **γ = 0.99**: a ~100-step planning horizon, needed for board-filling behaviour
  - Learning rate decay: 0.002 → 0.001 (ep 500) → 0.0005 (ep 800)

- **Training reward (shield mode, stationary)**: `+1` food, `+10` win, `−1` death, `−0.01` per step, plus potential-based shaping toward the food. Kept fixed (no curriculum-scaled multipliers) so the learning target doesn't move under the network. Episodes end on a **starvation** cap (steps since last food), not a total-step cap, so the snake can experience long games.

> **Note:** the earlier code advertised Prioritized Experience Replay, but the replay buffer is uniform — that claim has been removed rather than left inaccurate. The curriculum system still schedules exploration/learning-rate, but the reward itself is now the stationary scheme above.

### Training UI Features

**Comprehensive GUI for Training Control:**

1. **Training Controls**
   - Episode count configuration (100-10000)
   - Learning rate adjustment (with spinbox for precision)
   - Batch size selection (32-512)
   - Save interval configuration
   - GPU/CPU device selection with CUDA check
   - Start/Stop training with progress monitoring

2. **Model Management**
   - **Episode Continuation**: Resume training from last checkpoint without restarting episode count
   - **Model Versioning**: Create numbered models (snake_enhanced_dqn_1.pth, _2.pth, etc.)
   - Auto-detection of next available model number
   - Browse and load existing models
   - View model statistics and training history
   - Delete old models

3. **Visualization Tabs**
   - **Training Performance**: Real-time score and loss graphs with dual axes
   - **Model Visualization**: 
     - Network architecture diagrams
     - Feature importance analysis (gradient-based)
     - Live state analysis with Q-value breakdown and radar charts
   - **Training Log**: Separate logs for training events and system messages

4. **Model Browser** (In Main Game)
   - Cycle through all available .pth models
   - Automatic model type detection (Enhanced vs Standard)
   - Live model switching from menu

### Debug Mode

Press **'G'** during gameplay to toggle debug overlay:
- Current Q-values for all actions
- Danger state visualization (immediate and extended)
- Food direction indicators
- A* path suggestions (Enhanced DQN only)
- Current action and state summary

## 📋 Requirements

- Python 3.11 or higher
- PyTorch (with CUDA support for GPU training)
- Pygame
- Matplotlib
- NumPy
- Tkinter (for training UI)
- psutil (for memory monitoring)

## Installation

1. Clone the repository and navigate to the Snake Game directory:

```bash
cd "Snake Game/advanced_snake"
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision pygame matplotlib numpy psutil
```

## Usage

### Playing the Game

```bash
# From repository root
python play_snake.py

# Or directly
cd "Snake Game/advanced_snake"
python main.py
```

**Menu Controls:**
- **W/UP/S/DOWN**: Navigate menu
- **ENTER**: Select option / Cycle settings
- Menu options cycle through available values

**In-Game Controls:**
- **WASD / Arrow Keys**: Move snake (Manual mode)
- **P**: Pause/Resume
- **R**: Reset game
- **G**: Toggle debug mode (shows Q-values, danger states)
- **ESC**: Return to menu

### Training AI Models

#### **Using Training UI (Recommended)**

```bash
# From repository root
python train_snake.py

# Or directly
cd "Snake Game/advanced_snake"
python training_ui.py
```

**Training Steps:**
1. Select model type (Enhanced DQN recommended)
2. Set episode count (1000+ recommended for good performance)
3. Configure hyperparameters:
   - Learning Rate: 0.002 (default, decays automatically)
   - Batch Size: 64 (balance speed/stability)
   - Model Number: Leave empty for default, or specify for versioning
4. Check "Start from New Model" to train fresh, or uncheck to continue
5. Click "Start Training"
6. Monitor real-time graphs in "Training Performance" tab
7. Analyze features in "Model Visualization" tab

#### **Command-Line Training**

**Enhanced DQN (Recommended):**
```bash
cd "Snake Game/advanced_snake"

# Train new model
python train_enhanced.py --episodes 1000 --new-model

# Continue training from checkpoint
python train_enhanced.py --episodes 500

# Create numbered model version
python train_enhanced.py --episodes 1000 --model-number 1

# Custom hyperparameters
python train_enhanced.py --episodes 2000 --learning-rate 0.001 --batch-size 128 --model-number 2
```

**Original DQN:**
```bash
cd "Snake Game/advanced_snake"

# Train new model
python headless_training.py --episodes 1000 --new-model

# Continue training
python headless_training.py --episodes 500 --learning-rate 0.001
```

#### **Episode Continuation**

Training automatically continues from the last checkpoint:
- Episode count continues from where you left off
- Example: Train 100 episodes, stop, resume for 50 more → shows "Episode 101-150"
- History is preserved in `*_history.json` files

#### **Model Versioning**

Create multiple model versions for comparison:
```bash
# Create model #1
python train_enhanced.py --episodes 500 --model-number 1

# Create model #2 with different settings
python train_enhanced.py --episodes 1000 --learning-rate 0.001 --model-number 2

# Create model #3
python train_enhanced.py --episodes 2000 --model-number 3
```

Models are saved as:
- `models/snake_enhanced_dqn_1.pth` + `snake_enhanced_dqn_1_history.json`
- `models/snake_enhanced_dqn_2.pth` + `snake_enhanced_dqn_2_history.json`
- etc.

### Using the Model Browser

In the main game menu:
1. Navigate to "DQN Model: Browse"
2. Press **ENTER** to cycle through all available models
3. Current model name is displayed
4. Select "Start Game" and choose "DQN" or "Enhanced DQN" mode
5. The selected model will be used

## Training Tips

### For Best Results:

1. **Start with Enhanced DQN**: Better architecture and features
2. **Train for 1000+ episodes**: Allows curriculum progression
3. **Use default learning rate (0.002)**: Proven to work well with automatic decay
4. **Monitor the graphs**: 
   - Score should trend upward
   - Loss should decrease then stabilize
   - Running average is more important than individual episode scores
5. **Let curriculum stages complete**: 
   - Stage 0→1: ~100-150 episodes
   - Stage 1→2: ~200-300 episodes
   - Stage 2→3: ~400-600 episodes
   - Stage 3→4: ~800-1000 episodes

### Troubleshooting:

**Low scores after training:**
- Train longer (2000+ episodes)
- Check if model is actually loading (look for "Loaded model" message)
- Try different model numbers
- Verify CUDA is working for faster training

**Training too slow:**
- Use GPU if available (check with CUDA Check button in UI)
- Increase batch size (128 or 256)
- Disable real-time graphs option
- Use command-line training instead of UI

**Model not improving:**
- Reset and train from scratch with `--new-model`
- Try different learning rates (0.001 or 0.003)
- Increase episodes to 2000+
- Check that curriculum stages are progressing

## Model Architecture

### Enhanced DQN Network

```
Input Layer (34 features)
    ↓
Feature Layer 1 (256 nodes, ReLU, Dropout 0.2)
    ↓
Feature Layer 2 (256 nodes, ReLU, Dropout 0.2)
    ↓
Split into two streams:

Value Stream (128 nodes)          Advantage Stream (128 nodes)
    ↓                                      ↓
State Value (1 node)              Advantage per Action (3 nodes)
    ↓                                      ↓
         Combined (Q-values for 3 relative actions)
```

**Actions are relative:** `0 = turn right`, `1 = straight`, `2 = turn left`.

**Inference note:** the network is switched to `eval()` mode during action selection so **Dropout is disabled** — previously it was left in `train()` mode, which perturbed every decision with 20% of neurons randomly dropped. This is now fixed.

## Project Structure

```
Snake Game/
├── play_snake.py              # Main game launcher
├── train_snake.py             # Training UI launcher
└── advanced_snake/
    ├── main.py                # Game entry point
    ├── game_engine.py         # Core game logic
    ├── constants.py           # Configuration
    ├── algorithms.py          # A*, Dijkstra, Hamiltonian, DHCR implementations
    ├── safety.py              # Survival shield + Hamiltonian cycle backbone (the win layer)
    ├── q_learning.py          # Q-learning agent
    ├── advanced_dqn.py        # Original DQN agent
    ├── enhanced_dqn.py        # Enhanced DQN agent (34 features, shield-integrated)
    ├── train_enhanced.py      # Enhanced DQN training script
    ├── headless_training.py   # Original DQN training script
    ├── training_ui.py         # Training GUI
    ├── dqn_training.py        # In-game training interface
    ├── training.py            # Q-learning training interface
    └── models/                # Saved model directory
        ├── snake_enhanced_dqn.pth
        ├── snake_enhanced_dqn_history.json
        ├── snake_enhanced_dqn_1.pth
        └── ...
```

## Learning Resources

### Understanding the Features

**Danger Detection (13 features):**
- Immediate danger in 3 directions
- Extended danger (2 steps ahead)
- Trap detection (enclosed spaces)
- Body proximity (how close snake body is)
- Wall distances

**Food Information (6 features):**
- Direction to food (up/down/left/right)
- X and Y distances to food

**Navigation (12 features):**
- Current movement direction (one-hot)
- Available space in each direction
- Snake length
- Tail position and distance
- Moves until tail clears current position

**A* Hints (3 features - Enhanced only):**
- A* suggests straight/right/left

### Training Metrics

- **Score**: Food collected in an episode
- **Steps**: Number of moves before game over
- **Best Score**: Highest score achieved
- **Average Score**: Rolling average over last 100 episodes
- **Epsilon**: Current exploration rate
- **Curriculum Stage**: Current learning stage (0-4)
- **A* Guidance Probability**: How often A* hints are used for rewards

## Known Issues & Notes

1. **CUDA Availability**: If PyTorch can't detect GPU, run `check_cuda.py` in advanced_snake directory. Note: the network is tiny, so training is mostly **CPU-bound** (the per-step A*/shield logic dominates) — GPU utilization will look low, which is expected.
2. **Training UI Memory**: Long training sessions may consume significant RAM; restart UI if slow
3. **Model Compatibility**: Enhanced DQN models are NOT compatible with Original DQN mode (and vice versa)
4. **Cycle backbone alignment**: when you start a DQN game in the default win mode, the snake **snaps to the cycle's start corner** on the first frame — this is the alignment step, not a bug.
5. **Watching a full win**: a perfect 30×30 game is ~189,000 moves; the AI computes it in ~15s, but on-screen it's gated by the render speed — use the fastest speed setting to watch it complete.

## Version History

### v4.0 (Current) — "The snake wins"
- **Survival shield + Hamiltonian cycle backbone** (`safety.py`): the trained snake now fills the board (score 8960) — a guaranteed perfect game, verified 8×8 → 30×30.
- Loop-breaker (hunger-triggered shortest-safe-path) so the snake never circles without eating.
- **Bug fixes:** network now runs in `eval()` mode at inference (Dropout was corrupting every decision); real **Double DQN** target; **Huber** loss; **γ 0.95 → 0.99**; removed dead/obfuscated watermark code.
- **Fixed a win-time crash:** the "perfect game" message printed an emoji that crashed the Windows console at the exact moment of winning; win condition is now grid-agnostic ASCII with a `won` flag.
- Training reworked: stationary shielded reward, starvation-based (not total-step) episode cap.
- Corrected docs: removed the never-implemented Prioritized Experience Replay claim.

### v3.0
- Added Enhanced DQN with 34 features
- Implemented curriculum learning
- Added A* reward shaping
- Episode continuation support
- Model versioning system
- Model browser in main game
- Feature importance analysis
- Live state visualization
- Network architecture diagrams

### v2.0
- Added Original DQN with 11 features
- Training UI with real-time graphs
- Double DQN and Dueling architecture
- Prioritized Experience Replay

### v1.0
- Basic Snake game with Manual, A*, Dijkstra modes
- Q-learning agent

## Contributing

Contributions are welcome! Areas for improvement:
- Additional curriculum stages
- Hyperparameter optimization
- New feature engineering
- Performance benchmarking
- Documentation improvements

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Deep Q-Learning paper: Mnih et al. (2015)
- Double DQN: van Hasselt et al. (2015)
- Dueling DQN: Wang et al. (2016)
- Hamiltonian-cycle-with-shortcuts strategy for a guaranteed Snake win (John Tapsell / AlphaPhoenix DHCR)

---

**Happy Snake Gaming and Training! 🐍**