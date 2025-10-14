"""
Snake AI Training Launcher
===========================
Launch the training UI for DQN and Enhanced DQN agents.

This script opens the graphical training interface where you can:
- Train enhanced DQN agents (34 features with A* reward shaping)
- Monitor real-time training graphs
- Adjust hyperparameters
- Continue from checkpoints
- Compare model performance

Usage:
    python train_snake.py

Training Models:
- Enhanced DQN: 34-feature state with curriculum learning and A* guidance

Features:
- Real-time performance graphs
- Configurable hyperparameters (learning rate, epsilon, batch size)
- GPU acceleration support
- Checkpoint save/load
- Episode statistics and logs
"""

import sys
import os

# Add the advanced_snake directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
advanced_snake_dir = os.path.join(current_dir, 'advanced_snake')
sys.path.insert(0, advanced_snake_dir)

# Change working directory to advanced_snake
os.chdir(advanced_snake_dir)

# Import and run the training UI
from advanced_snake.training_ui import main as run_training_ui

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 SNAKE AI TRAINING INTERFACE")
    print("=" * 70)
    print("\nAvailable Models:")
    print("  • Enhanced DQN - 34 features, A* reward shaping, curriculum learning")
    print("\nTraining Features:")
    print("  • Real-time performance graphs")
    print("  • Configurable hyperparameters")
    print("  • GPU acceleration (if available)")
    print("  • Checkpoint save/load")
    print("  • Curriculum learning stages: [20, 50, 100, 200] ✨ LOWERED!")
    print("  • A* reward shaping: 0.5 → 0.35 → 0.20 → 0.10 → 0.0")
    print("  • Food reward: DOUBLED (20 points) 🍎")
    print("  • Epsilon caps: LOWERED [0.2, 0.15, 0.1, 0.05] ⚡")
    print("  • Learning rate decay: Episode 500 & 800 📉")
    print("\n✅ ALL PERFORMANCE FIXES APPLIED!")
    print("\nLaunching training interface...\n")
    print("=" * 70)
    
    try:
        run_training_ui()
    except KeyboardInterrupt:
        print("\n\nTraining interface interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running training interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
