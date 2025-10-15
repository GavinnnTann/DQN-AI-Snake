"""
Snake Game Launcher
===================
Simple launcher to play the Snake game.

This script launches the main Snake game with a GUI.
You can play manually or watch AI agents play.

Usage:
    python play_snake.py

Features:
- Manual gameplay (WASD or Arrow keys)
- Watch A* algorithm play
- Watch trained DQN agents play
- Pause/Resume with 'P'
- Reset with 'R'
"""

import sys
import os

# Add the advanced_snake directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
advanced_snake_dir = os.path.join(current_dir, 'advanced_snake')
sys.path.insert(0, advanced_snake_dir)

# Change working directory to advanced_snake
os.chdir(advanced_snake_dir)

# Import and run the main game
from advanced_snake.main import SnakeGame

if __name__ == "__main__":
    print("=" * 70)
    print("🐍 SNAKE GAME")
    print("=" * 70)
    print("\nControls:")
    print("  • WASD or Arrow Keys - Move snake")
    print("  • P - Pause/Resume")
    print("  • R - Reset game")
    print("  • M - Switch to menu")
    print("  • ESC - Exit")
    print("\nGame Modes:")
    print("  • Manual Play")
    print("  • Watch A* Algorithm")
    print("  • Watch Dijkstra's Algorithm")
    print("  • Watch Hamiltonian Cycle (Guaranteed Win!)")
    print("  • Watch DHCR (Smart Hamiltonian with Shortcuts)")
    print("  • Watch Q-Learning AI")
    print("  • Watch DQN AI")
    print("\nStarting game...\n")
    print("=" * 70)
    
    try:
        game = SnakeGame()
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running game: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
