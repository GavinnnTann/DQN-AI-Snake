"""
Q-Learning Training Script for Snake Game
==========================================
Headless training for Q-Learning agent with progress tracking and statistics export.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from collections import deque

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from q_learning import SnakeQLearningAgent
from constants import *


def train_qlearning(episodes=1000, save_interval=100, learning_rate=None, batch_size=None):
    """
    Train Q-Learning agent.
    
    Args:
        episodes: Number of episodes to train
        save_interval: Save model every N episodes
        learning_rate: Override alpha (learning rate) if provided
        batch_size: Override batch size for experience replay if provided
    """
    print("=" * 70)
    print("Q-LEARNING TRAINING - SNAKE GAME")
    print("=" * 70)
    print(f"Training Configuration:")
    print(f"  • Episodes: {episodes}")
    print(f"  • Save Interval: {save_interval}")
    print(f"  • Learning Rate (alpha): {learning_rate if learning_rate else QLEARNING_ALPHA}")
    print(f"  • Batch Size: {batch_size if batch_size else QLEARNING_BATCH_SIZE}")
    print(f"  • Discount Factor (gamma): {QLEARNING_GAMMA}")
    print(f"  • Initial Epsilon: {QLEARNING_EPSILON}")
    print(f"  • Epsilon Decay: {QLEARNING_EPSILON_DECAY}")
    print(f"  • Min Epsilon: {QLEARNING_EPSILON_MIN}")
    print("=" * 70)
    
    # Initialize game engine and agent
    game_engine = GameEngine()
    agent = SnakeQLearningAgent(game_engine)
    
    # Override parameters if specified
    if learning_rate is not None:
        agent.alpha = learning_rate
        print(f"[OK] Learning rate overridden to: {learning_rate}")
    
    if batch_size is not None:
        # Note: Q-Learning uses this for experience replay batch size
        print(f"[OK] Batch size set to: {batch_size}")
    else:
        batch_size = QLEARNING_BATCH_SIZE
    
    # Check if we should continue from existing model
    model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
    starting_episode = 0
    
    if os.path.exists(model_path):
        try:
            if agent.load_model(model_path):
                print(f"[OK] Loaded existing Q-Learning model from {model_path}")
                print(f"    • Q-table size: {len(agent.q_table)} states")
                print(f"    • Current epsilon: {agent.epsilon:.4f}")
                
                # Try to get episode count from stats
                if agent.stats and 'scores' in agent.stats:
                    starting_episode = len(agent.stats['scores'])
                    print(f"    • Resuming from episode {starting_episode}")
        except Exception as e:
            print(f"[WARNING] Could not load existing model: {e}")
            print("[INFO] Starting with fresh Q-table")
    else:
        print("[INFO] No existing model found. Starting with fresh Q-table")
    
    # Training statistics
    training_stats = {
        'episodes': [],
        'scores': [],
        'steps': [],
        'epsilon': [],
        'avg_reward': [],
        'q_table_size': [],
        'running_avg': []
    }
    
    # Running averages
    recent_scores = deque(maxlen=100)
    best_score = 0
    best_avg = 0
    running_avg = 0.0  # Initialize to avoid UnboundLocalError
    
    print("\n[INFO] Starting training...\n")
    
    try:
        for episode in range(starting_episode, starting_episode + episodes):
            # Reset game
            game_engine.reset_game()
            episode_reward = 0
            steps = 0
            
            # Play one episode
            while not game_engine.game_over:
                # Store old state
                old_state = agent.get_state()
                old_score = game_engine.score
                
                # Get action and execute
                action = agent.get_action(old_state)
                game_engine.set_direction_from_algorithm(action)
                game_engine.move_snake()
                
                # Get new state and calculate reward
                new_state = agent.get_state()
                done = game_engine.game_over
                ate_food = game_engine.score > old_score
                
                reward = agent.calculate_reward(old_state, new_state, action, done, ate_food)
                episode_reward += reward
                
                # Store experience
                agent.remember(old_state, action, reward, new_state, done)
                
                # Train with experience replay
                if len(agent.memory) >= batch_size:
                    agent.replay(batch_size)
                
                steps += 1
                
                # Prevent infinite loops
                if steps > 1000:
                    break
            
            # Update statistics
            score = game_engine.score
            recent_scores.append(score)
            running_avg = sum(recent_scores) / len(recent_scores)
            
            training_stats['episodes'].append(episode + 1)
            training_stats['scores'].append(score)
            training_stats['steps'].append(steps)
            training_stats['epsilon'].append(agent.epsilon)
            training_stats['avg_reward'].append(episode_reward / max(steps, 1))
            training_stats['q_table_size'].append(len(agent.q_table))
            training_stats['running_avg'].append(running_avg)
            
            # Track best performance
            if score > best_score:
                best_score = score
            if running_avg > best_avg:
                best_avg = running_avg
            
            # Progress output
            if (episode + 1) % 10 == 0 or episode == starting_episode:
                print(f"Episode {episode + 1:4d} | "
                      f"Score: {score:3d} | "
                      f"Steps: {steps:4d} | "
                      f"Avg: {running_avg:6.2f} | "
                      f"epsilon: {agent.epsilon:.4f} | "
                      f"Q-states: {len(agent.q_table):5d} | "
                      f"Best: {best_score:3d}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                agent.save_model(model_path)
                
                # Also save training stats
                stats_path = os.path.join(QMODEL_DIR, "qlearning_training_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(training_stats, f, indent=2)
                
                print(f"\n[SAVE] Model and stats saved at episode {episode + 1}")
                print(f"       Best Score: {best_score} | Best Avg: {best_avg:.2f}")
                print(f"       Q-table size: {len(agent.q_table)} states\n")
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Training interrupted by user.")
    
    except Exception as e:
        print(f"\n\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final save
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        agent.save_model(model_path)
        print(f"[SAVE] Final model saved to: {model_path}")
        
        # Save final statistics
        stats_path = os.path.join(QMODEL_DIR, "qlearning_training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        print(f"[SAVE] Training statistics saved to: {stats_path}")
        
        # Summary
        print(f"\nTraining Summary:")
        print(f"  • Total Episodes: {len(training_stats['scores'])}")
        print(f"  • Best Score: {best_score}")
        print(f"  • Best Avg (last 100): {best_avg:.2f}")
        print(f"  • Final Avg (last 100): {running_avg:.2f}")
        print(f"  • Final Epsilon: {agent.epsilon:.4f}")
        print(f"  • Q-table Size: {len(agent.q_table)} states")
        print("=" * 70)


def main():
    """Parse command line arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Q-Learning agent for Snake Game')
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save model every N episodes (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (alpha) - overrides default')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for experience replay - overrides default')
    
    args = parser.parse_args()
    
    # Start training
    train_qlearning(
        episodes=args.episodes,
        save_interval=args.save_interval,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
