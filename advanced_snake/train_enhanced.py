"""
Training script for Enhanced DQN Agent with curriculum learning and A* guidance.

Usage:
    python train_enhanced.py --episodes 1000 --new-model

This will train an enhanced agent with:
- 31-feature state representation (vs original 11)
- A* guided exploration in early stages
- Curriculum learning with progressive difficulty
- Enhanced reward shaping
- Better spatial awareness
"""

import argparse
import os
import time
import json
import torch
import numpy as np
from datetime import datetime

from constants import *
from game_engine import GameEngine
from enhanced_dqn import EnhancedDQNAgent


def train_enhanced_dqn(episodes=1000, use_existing=True, save_interval=50, learning_rate=None, batch_size=None, model_number=None, train_every=4, use_shield=True):
    """
    Train the enhanced DQN agent.

    Args:
        episodes: Number of training episodes
        use_existing: Whether to load existing model
        save_interval: Save model every N episodes
        learning_rate: Learning rate for optimizer (if None, uses value from constants.py)
        batch_size: Batch size for training (if None, uses value from constants.py)
        model_number: Specific model number to save/load (e.g., 1, 2, 3). If None, uses default name.
        train_every: Run a gradient update every N environment steps (higher = faster,
                     fewer updates per step of experience). 4 is the classic DQN value.
        use_shield: If True, explore inside the survival shield (safe but LONG episodes).
                    If False, plain epsilon-greedy - the snake dies early, so episodes are
                    much shorter and training is far faster. The play-time cycle backbone
                    guarantees the win regardless, so shield-free training is usually fine.
    """
    print("\n" + "="*70)
    print("ENHANCED DQN TRAINING WITH A* GUIDANCE")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"State Features: 34 (includes A* path hints in state)")
    print(f"Curriculum Stages: {[25, 60, 120, 250]} score thresholds (UPDATED)")
    print(f"A* Guidance: Via reward shaping, not action override")
    print(f"A* Reward Weight: {0.5} at Stage 0, reduces with curriculum")
    print("="*70 + "\n")
    
    # Setup
    os.makedirs(QMODEL_DIR, exist_ok=True)
    print(f"[OK] Models directory: {os.path.abspath(QMODEL_DIR)}")
    
    game_engine = GameEngine()
    agent = EnhancedDQNAgent(game_engine)

    # Train in SHIELD mode, not the cycle backbone. The backbone plays a full
    # ~200k-step winning game every episode (infeasible to train on) and it
    # guarantees the win regardless of the network anyway. Training instead
    # sharpens the network's food-seeking (used at play time both for the
    # non-guaranteed "pure shield" mode and to pick among safe backbone shortcuts).
    agent.use_cycle_backbone = False

    # Speed controls.
    agent.use_safety_shield = use_shield
    print(f"[OK] Safety shield during training: {'ON (safe exploration, longer episodes)' if use_shield else 'OFF (plain e-greedy, short/fast episodes)'}")
    print(f"[OK] Gradient update every {train_every} step(s)")

    # Track starting episode for continuation
    start_episode = 1
    
    # Override learning rate if specified
    if learning_rate is not None:
        agent.learning_rate = learning_rate
        # Update optimizer with new learning rate
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = learning_rate
        print(f"[OK] Learning rate set to: {learning_rate}")
    else:
        print(f"[OK] Using default learning rate: {agent.learning_rate}")
    
    # Override batch size if specified
    if batch_size is not None:
        agent.batch_size = batch_size
        print(f"[OK] Batch size set to: {batch_size}")
    else:
        print(f"[OK] Using default batch size: {agent.batch_size}")
    
    # Determine model filename
    if model_number is not None:
        model_filename = f"snake_enhanced_dqn_{model_number}.pth"
        history_filename = f"snake_enhanced_dqn_{model_number}_history.json"
    else:
        model_filename = "snake_enhanced_dqn.pth"
        history_filename = "snake_enhanced_dqn_history.json"
    
    model_path = os.path.join(QMODEL_DIR, model_filename)
    history_path = os.path.join(QMODEL_DIR, history_filename)
    print(f"[OK] Model path: {os.path.abspath(model_path)}")
    
    # Load existing model if requested
    if use_existing and os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            print(f"[OK] Loaded existing enhanced model from {model_path}")
            
            # Try to load training history to continue from last episode
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history_data = json.load(f)
                        start_episode = history_data.get('episodes_completed', 0) + 1
                        
                        # Restore curriculum stage and related parameters
                        if 'curriculum_stage' in history_data:
                            agent.curriculum_stage = history_data['curriculum_stage']
                            print(f"[OK] Restored curriculum stage: {agent.curriculum_stage}")
                            
                            # Set appropriate A* guidance probability for the stage
                            astar_probs = {0: 0.5, 1: 0.35, 2: 0.20, 3: 0.10, 4: 0.0}
                            agent.astar_guidance_prob = astar_probs.get(agent.curriculum_stage, 0.0)
                            print(f"[OK] Set A* guidance probability: {agent.astar_guidance_prob}")
                        
                        # Restore epsilon if it was saved
                        if 'epsilon' in history_data:
                            agent.epsilon = history_data['epsilon']
                            print(f"[OK] Restored epsilon: {agent.epsilon:.4f}")
                        
                        print(f"[OK] Resuming from episode {start_episode}")
                except Exception as e:
                    print(f"[WARNING] Could not load history: {e}")
                    start_episode = 1
            else:
                print(f"[WARNING] No history file found, starting episode count from 1")
                start_episode = 1
            print()
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
            print("Starting with fresh model\n")
            start_episode = 1
    else:
        if not use_existing:
            print(f"[OK] Starting with fresh model (--new-model flag set)\n")
        else:
            print(f"[OK] No existing model found, starting with fresh model\n")
        start_episode = 1
    
    # Training statistics - Load from history if continuing
    scores = []
    running_avgs = []
    best_score = 0
    
    # BUGFIX: Load previous training history when continuing
    if use_existing and os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                # Restore training statistics
                scores = history_data.get('scores', [])
                running_avgs = history_data.get('running_avgs', [])
                best_score = history_data.get('best_score', 0)
                print(f"[OK] Loaded training history:")
                print(f"    - Previous episodes: {len(scores)}")
                print(f"    - Best score: {best_score}")
                print(f"    - Latest avg: {running_avgs[-1] if running_avgs else 0:.2f}")
                print()
        except Exception as e:
            print(f"[WARNING] Could not load training history for statistics: {e}")
            print("[INFO] Starting with fresh statistics\n")
    
    training_start = time.time()
    
    # Training loop - now continues from start_episode
    total_episodes = start_episode + episodes - 1
    for episode in range(start_episode, total_episodes + 1):
        # ADDED: Learning rate decay for stability
        # Reduce learning rate as training progresses to fine-tune weights
        if episode == 500:
            old_lr = agent.optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.5  # Halve the learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"\n[LEARNING RATE DECAY] Episode {episode}: {old_lr:.6f} -> {new_lr:.6f}")
        elif episode == 800:
            old_lr = agent.optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.5  # Halve again
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"\n[LEARNING RATE DECAY] Episode {episode}: {old_lr:.6f} -> {new_lr:.6f}")
        
        game_engine.reset_game()
        agent.on_new_game()   # reset shield state + align snake to cycle (win backbone)
        state = agent.get_state()
        episode_reward = 0
        steps = 0
        steps_since_food = 0
        episode_start = time.time()
        old_score = 0

        # Starvation limit: with the survival shield the snake can live almost
        # indefinitely, so we no longer cap TOTAL steps (that was a hard score
        # ceiling). Instead we end the episode only if it goes too long WITHOUT
        # eating, which means it is genuinely looping. Scale generously so the
        # endgame (where reaching food can require traversing most of the board)
        # is not cut short.
        area = GRID_WIDTH * GRID_HEIGHT
        starvation_limit = 2 * area

        while not game_engine.game_over:
            # Get old distance to food
            head = game_engine.snake[0]
            food = game_engine.food
            old_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])

            # Select and perform action (shielded)
            action = agent.select_action(state, training=True)
            agent.perform_action(action)

            # Get new state and reward
            new_state = agent.get_state()
            new_distance = abs(game_engine.snake[0][0] - food[0]) + abs(game_engine.snake[0][1] - food[1])
            reward = agent.calculate_reward_shielded(old_score, game_engine.game_over, old_distance, new_distance)

            # Store transition
            agent.memory.add(state, action, reward, new_state, game_engine.game_over)

            # Train the agent every `train_every` steps (optimize is ~70% of a
            # step's cost, so this is the single biggest wall-clock speedup).
            if steps % train_every == 0:
                loss = agent.optimize_model()

            # Track starvation (steps since last food)
            if game_engine.score > old_score:
                steps_since_food = 0
            else:
                steps_since_food += 1

            # Update state
            state = new_state
            old_score = game_engine.score
            episode_reward += reward
            steps += 1

            # End only on genuine looping (no food for a long time)
            if steps_since_food > starvation_limit:
                break
        
        # Episode finished
        score = game_engine.score
        scores.append(score)
        agent.update_curriculum(score, current_episode=episode)  # Update curriculum stage (pass episode for cooldown tracking)
        
        # ============================================================
        # PROGRESSIVE EPSILON DECAY (per episode)
        # ============================================================
        # Decay epsilon ONCE per episode with curriculum-based minimum
        # Different minimum epsilon per stage to maintain exploration
        # NOTE: Decay parameters are configured in constants.py
        # Edit STAGE_EPSILON_MINIMUMS and STAGE_EPSILON_DECAY to adjust
        stage_epsilon_min = STAGE_EPSILON_MINIMUMS.get(agent.curriculum_stage, 0.01)
        
        # PERFORMANCE FIX: Force epsilon back up if it dropped too low
        # This can happen if epsilon was saved at a low value or decay was too aggressive
        if agent.epsilon < stage_epsilon_min:
            print(f"[EPSILON FIX] Epsilon {agent.epsilon:.4f} below minimum {stage_epsilon_min:.4f}, correcting...")
            agent.epsilon = stage_epsilon_min
        
        # PERFORMANCE BOOST: Curriculum-adaptive epsilon decay
        # Faster decay at early stages for quicker exploitation
        epsilon_decay_rate = STAGE_EPSILON_DECAY.get(agent.curriculum_stage, 0.997)
        
        if agent.epsilon > stage_epsilon_min:
            agent.epsilon *= epsilon_decay_rate
        else:
            agent.epsilon = stage_epsilon_min  # Enforce minimum for current stage
        
        # ============================================================
        # PROGRESSIVE LEARNING RATE DECAY (per episode) - NEW!
        # ============================================================
        # Apply same decay strategy to learning rate for stability
        # NOTE: Decay parameters are configured in constants.py
        # Edit STAGE_LR_MINIMUMS and STAGE_LR_DECAY to adjust
        stage_lr_min = STAGE_LR_MINIMUMS.get(agent.curriculum_stage, 0.0002)
        
        # Decay rate slightly slower than epsilon (want to keep learning longer)
        lr_decay_rate = STAGE_LR_DECAY.get(agent.curriculum_stage, 0.9995)
        
        # Get current learning rate from optimizer
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        if current_lr > stage_lr_min:
            new_lr = current_lr * lr_decay_rate
            # Update optimizer learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr
            agent.learning_rate = new_lr
        else:
            # Enforce minimum
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = stage_lr_min
            agent.learning_rate = stage_lr_min
        
        # Calculate running average
        window = min(100, len(scores))
        running_avg = np.mean(scores[-window:])
        running_avgs.append(running_avg)
        
        # Update best score
        if score > best_score:
            best_score = score
        
        # Calculate episode time
        episode_time = time.time() - episode_start
        
        # Update target network periodically for stable Q-learning
        if episode % 10 == 0:
            agent.update_target_network()
            if episode % 50 == 0:  # Log every 50 episodes
                print(f"[TARGET NET] Updated target network at episode {episode}", flush=True)
        
        # Print progress with A* guidance info + learning rate
        print(f"Enhanced DQN Episode: {episode}/{total_episodes}, "
              f"Score: {score:.1f}, Steps: {steps}, "
              f"Best: {best_score:.1f}, Avg: {running_avg:.2f}, "
              f"Epsilon: {agent.epsilon:.4f}, LR: {agent.learning_rate:.5f}, "
              f"Curriculum: Stage {agent.curriculum_stage}, "
              f"A*: {agent.astar_guidance_prob:.2f}, "
              f"Time: {episode_time:.2f}s", flush=True)
        
        # Save model periodically
        if episode % save_interval == 0:
            try:
                agent.save_model(model_path)
                print(f"[SAVED] Model saved to {model_path}", flush=True)
            except Exception as e:
                print(f"[ERROR] Error saving model: {e}", flush=True)
            
            # Save training history
            try:
                history = {
                    'scores': scores,
                    'running_avgs': running_avgs,
                    'best_score': best_score,
                    'latest_avg_score': running_avg,
                    'episodes_completed': episode,
                    'training_time': time.time() - training_start,
                    'timestamp': time.time(),
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'curriculum_stage': agent.curriculum_stage,
                    'epsilon': agent.epsilon,  # Save current epsilon for resumption
                    'astar_guidance_prob': agent.astar_guidance_prob,  # Save A* guidance probability
                    'state_features': 34,
                    'model_type': 'Enhanced DQN with A* Reward Shaping'
                }
                
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                print(f"[SAVED] Training history saved to {history_path}", flush=True)
            except Exception as e:
                print(f"[ERROR] Error saving training history: {e}", flush=True)
    
    # Final save
    try:
        agent.save_model(model_path)
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total Episodes: {episodes}")
        print(f"Best Score: {best_score:.1f}")
        print(f"Final Average (100 ep): {running_avg:.2f}")
        print(f"Final Curriculum Stage: {agent.curriculum_stage}/4")
        print(f"Total Time: {(time.time() - training_start)/60:.1f} minutes")
        print(f"Model saved to: {model_path}")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Error in final save: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Enhanced DQN Agent for Snake Game')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--new-model', action='store_true', help='Start with a fresh model')
    parser.add_argument('--save-interval', type=int, default=50, help='Save model every N episodes')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training (32-512)')
    parser.add_argument('--model-number', type=int, default=None, help='Model number for saving (e.g., 1, 2, 3)')
    parser.add_argument('--train-every', type=int, default=4, help='Run a gradient update every N steps (default 4; higher = faster)')
    parser.add_argument('--no-shield', action='store_true', help='Disable the survival shield during training (much faster: episodes end on death instead of surviving thousands of steps)')

    # Stuck detection parameters
    parser.add_argument('--enable-stuck-detection', action='store_true', help='Enable stuck detection (default from constants.py)')
    parser.add_argument('--disable-stuck-detection', action='store_true', help='Disable stuck detection')
    parser.add_argument('--stuck-sensitivity', type=int, default=None, help='Stuck counter threshold (1-10, default from constants.py)')
    parser.add_argument('--stuck-cooldown', type=int, default=None, help='Cooldown between boosts in episodes (default from constants.py)')
    parser.add_argument('--stuck-boost', type=float, default=None, help='Epsilon boost amount (default from constants.py)')
    parser.add_argument('--stuck-improvement', type=float, default=None, help='Improvement threshold (default from constants.py)')
    
    args = parser.parse_args()
    
    # Apply stuck detection settings to constants (will be used by agent)
    if args.disable_stuck_detection:
        import constants
        constants.ENABLE_STUCK_DETECTION = False
        print("[CONFIG] Stuck detection: DISABLED")
    elif args.enable_stuck_detection:
        import constants
        constants.ENABLE_STUCK_DETECTION = True
        if args.stuck_sensitivity is not None:
            constants.STUCK_COUNTER_THRESHOLD = args.stuck_sensitivity
        if args.stuck_cooldown is not None:
            constants.STUCK_BOOST_COOLDOWN = args.stuck_cooldown
        if args.stuck_boost is not None:
            constants.STUCK_EPSILON_BOOST = args.stuck_boost
        if args.stuck_improvement is not None:
            constants.STUCK_IMPROVEMENT_THRESHOLD = args.stuck_improvement
        print(f"[CONFIG] Stuck detection: ENABLED")
        print(f"  • Sensitivity: {constants.STUCK_COUNTER_THRESHOLD} checks ({constants.STUCK_COUNTER_THRESHOLD * 50} episodes)")
        print(f"  • Cooldown: {constants.STUCK_BOOST_COOLDOWN} episodes")
        print(f"  • Boost: +{constants.STUCK_EPSILON_BOOST}")
        print(f"  • Min improvement: {constants.STUCK_IMPROVEMENT_THRESHOLD} points")
    
    train_enhanced_dqn(
        episodes=args.episodes,
        use_existing=not args.new_model,
        save_interval=args.save_interval,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model_number=args.model_number,
        train_every=args.train_every,
        use_shield=not args.no_shield
    )
