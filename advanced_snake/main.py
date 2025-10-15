"""
Main module for the Snake Game.
Integrates the game engine, algorithms, and UI components.
Uses pygame exclusively for rendering and input handling.
"""

import pygame
import sys
import os
import torch  # For DQN debugging
from pygame.locals import KEYDOWN, K_w, K_a, K_s, K_d, K_p, K_r, K_t, K_g, K_ESCAPE, K_RETURN, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_m

# Game version and attribution - Global Author Variable Identification Number: Tan
__author__ = "Gavin Tan"
__version__ = "2.0"

# Import custom modules
from constants import *
from game_engine import GameEngine
from algorithms import SnakeAlgorithms
from q_learning import SnakeQLearningAgent
from enhanced_dqn import EnhancedDQNAgent  # Enhanced DQN with A* and curriculum learning

class SnakeGame:
    def __init__(self):
        """Initialize the Snake Game."""
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)
        
        # Set up display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Ensure models directory exists
        os.makedirs(QMODEL_DIR, exist_ok=True)
        
        # Initialize game components
        self.game_engine = GameEngine()
        self.algorithms = SnakeAlgorithms(self.game_engine)
        
        # Create Q-learning agent if model exists
        self.q_agent = None
        model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
        if os.path.exists(model_path):
            self.q_agent = SnakeQLearningAgent(self.game_engine)
            self.q_agent.load_model(model_path)
            print(f"Loaded Q-learning model from {model_path}")
            
        # Create DQN agent if model exists
        self.dqn_agent = None
        self.current_dqn_model_path = None  # Track which model is loaded
        
        # Scan for all available DQN models
        self.available_dqn_models = self.scan_dqn_models()
        self.selected_model_index = 0  # Index of currently selected model
        
        # Load the first available model
        if self.available_dqn_models:
            self.load_dqn_model(0)  # Load first model
        
        # Game variables
        self.running = True
        self.game_state = STATE_MENU
        self.current_mode = MANUAL_MODE
        self.current_speed = FRAME_RATES[DEFAULT_SPEED]
        
        # Highscore tracking
        self.highscore = self.load_highscore()
        self.highscore_file = os.path.join(QMODEL_DIR, "highscore.txt")
        
        # Debug variables for DQN visualization
        self.debug_mode = False  # Press 'D' to toggle
        self.last_q_values = None
        self.last_action = None
        self.last_state_summary = None
        
        # Menu variables
        self.menu_options = ["Mode: " + MANUAL_MODE, "Speed: " + DEFAULT_SPEED, 
                             "DQN Model: Browse", "Start Game"]
        self.selected_option = 0
        
        # Mode selection variables
        self.mode_index = 0
        self.speed_index = list(FRAME_RATES.keys()).index(DEFAULT_SPEED)
        
        # Check if models are available and update game modes
        self.update_available_modes()
        
        # Update menu to show current model
        self.update_model_menu_option()
    
    def scan_dqn_models(self):
        """Scan for all available DQN models in the models directory."""
        import glob
        models = []
        
        # Find all .pth model files
        model_pattern = os.path.join(QMODEL_DIR, "*.pth")
        model_files = glob.glob(model_pattern)
        
        for model_path in sorted(model_files):
            filename = os.path.basename(model_path)
            
            # All DQN models use EnhancedDQNAgent architecture
            # Determine display type from filename
            if "curriculum" in filename.lower():
                model_type = "Enhanced (Curriculum)"
            elif "stable" in filename.lower():
                model_type = "Enhanced (Stable)"
            elif "enhanced" in filename.lower():
                model_type = "Enhanced DQN"
            else:
                model_type = "Enhanced DQN"
            
            models.append({
                'path': model_path,
                'filename': filename,
                'type': model_type,
                'agent_class': EnhancedDQNAgent  # All use EnhancedDQNAgent
            })
        
        if models:
            print(f"Found {len(models)} DQN model(s)")
        else:
            print("No DQN models found")
        
        # Model integrity verification - checksum GAV1N_TAN_2025
        return models
    
    def load_dqn_model(self, model_index):
        """Load a specific DQN model by index."""
        if not self.available_dqn_models or model_index >= len(self.available_dqn_models):
            return False
        
        model_info = self.available_dqn_models[model_index]
        model_path = model_info['path']
        agent_class = model_info['agent_class']
        
        try:
            # BUGFIX: PyTorch 2.6+ requires safe_globals for numpy types
            # This must be done BEFORE creating the agent (which calls load_model)
            import numpy as np
            if hasattr(torch.serialization, 'add_safe_globals'):
                # Directly add the specific numpy dtype classes that fail
                safe_types = []
                
                # Add np.dtype base class
                safe_types.append(np.dtype)
                
                # Add specific dtype classes (numpy 2.0+)
                if hasattr(np, 'dtypes'):
                    try:
                        safe_types.append(np.dtypes.Float64DType)
                        safe_types.append(np.dtypes.Float32DType)
                        safe_types.append(np.dtypes.Int64DType)
                        safe_types.append(np.dtypes.Int32DType)
                    except AttributeError:
                        pass
                
                # Add multiarray.scalar (try both namespaces)
                try:
                    safe_types.append(np._core.multiarray.scalar)
                except AttributeError:
                    try:
                        safe_types.append(np.core.multiarray.scalar)
                    except AttributeError:
                        pass
                
                # Register all types
                torch.serialization.add_safe_globals(safe_types)
            
            self.dqn_agent = agent_class(self.game_engine)
            # Load the model - the method only accepts filepath
            success = self.dqn_agent.load_model(model_path)
            if success:
                # Set epsilon to 0 for pure exploitation during gameplay
                self.dqn_agent.epsilon = 0.0
                self.current_dqn_model_path = model_path
                self.selected_model_index = model_index
                print(f"✅ Loaded {model_info['type']} DQN: {model_info['filename']}")
                print(f"   Mode: Gameplay (epsilon=0.0, pure exploitation)")
                return True
            else:
                print(f"❌ Failed to load {model_info['filename']}")
                return False
        except Exception as e:
            print(f"Error loading {model_info['filename']}: {e}")
            self.dqn_agent = None
            return False
    
    def load_highscore(self):
        """Load highscore from file."""
        highscore_file = os.path.join(QMODEL_DIR, "highscore.txt")
        try:
            if os.path.exists(highscore_file):
                with open(highscore_file, 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return 0
    
    def save_highscore(self, score):
        """Save highscore to file if it's a new record."""
        if score > self.highscore:
            self.highscore = score
            try:
                with open(self.highscore_file, 'w') as f:
                    f.write(str(score))
                print(f"🎉 NEW HIGHSCORE: {score}!")
                return True
            except Exception as e:
                print(f"Error saving highscore: {e}")
        return False
    
    def print_final_game_state(self):
        """Print the final game state to terminal when game ends."""
        print("\n" + "="*60)
        print("🎮 GAME OVER - Final State")
        print("="*60)
        
        # Basic game info
        print(f"Mode: {self.current_mode}")
        print(f"Final Score: {self.game_engine.score}")
        print(f"Snake Length: {len(self.game_engine.snake)}")
        print(f"Food Eaten: {self.game_engine.score // 10}")  # Each food = 10 points
        
        # Convert deque to list for slicing
        snake_list = list(self.game_engine.snake)
        
        # Snake position details
        head = snake_list[0]
        print(f"\nSnake Head Position: ({head[0]}, {head[1]})")
        print(f"Direction: {self.direction_to_string(self.game_engine.direction)}")
        
        # Food position
        food = self.game_engine.food
        print(f"Food Position: ({food[0]}, {food[1]})")
        
        # Calculate distance to food
        distance_to_food = abs(head[0] - food[0]) + abs(head[1] - food[1])
        print(f"Distance to Food: {distance_to_food} blocks")
        
        # Cause of death
        print(f"\n💀 Cause of Death:")
        if head[0] < 0 or head[0] >= GRID_WIDTH or head[1] < 0 or head[1] >= GRID_HEIGHT:
            print(f"   - Hit wall at boundary")
            if head[0] < 0:
                print(f"   - Left wall (x={head[0]})")
            elif head[0] >= GRID_WIDTH:
                print(f"   - Right wall (x={head[0]})")
            elif head[1] < 0:
                print(f"   - Top wall (y={head[1]})")
            else:
                print(f"   - Bottom wall (y={head[1]})")
        elif head in snake_list[1:]:
            collision_index = snake_list[1:].index(head) + 1
            print(f"   - Collided with own body")
            print(f"   - Hit segment #{collision_index} of {len(snake_list)}")
        
        # DQN specific debug info
        if self.current_mode == DQN_MODE and self.dqn_agent is not None:
            print(f"\n🤖 DQN Agent Info:")
            if self.last_action is not None:
                actions = ["Turn Right", "Straight", "Turn Left"]
                print(f"   Last Action: {actions[self.last_action]}")
            
            if self.last_q_values is not None:
                print(f"   Last Q-Values:")
                actions = ["Turn Right", "Straight", "Turn Left"]
                for i, (action, q_val) in enumerate(zip(actions, self.last_q_values)):
                    marker = " ←" if i == self.last_action else ""
                    print(f"      {action}: {q_val:.4f}{marker}")
            
            if self.last_state_summary is not None:
                print(f"   State at Death:")
                print(f"      Danger - Straight: {self.last_state_summary.get('danger_straight', 0) > 0.5}")
                print(f"      Danger - Right: {self.last_state_summary.get('danger_right', 0) > 0.5}")
                print(f"      Danger - Left: {self.last_state_summary.get('danger_left', 0) > 0.5}")
        
        # Snake body visualization (first 5 and last 5 segments for long snakes)
        print(f"\n🐍 Snake Body ({len(snake_list)} segments):")
        if len(snake_list) <= 10:
            for i, segment in enumerate(snake_list):
                marker = "HEAD" if i == 0 else f"#{i}"
                print(f"   [{marker}] ({segment[0]}, {segment[1]})")
        else:
            # Show first 5
            for i in range(5):
                segment = snake_list[i]
                marker = "HEAD" if i == 0 else f"#{i}"
                print(f"   [{marker}] ({segment[0]}, {segment[1]})")
            print(f"   ... ({len(snake_list) - 10} segments omitted) ...")
            # Show last 5
            for i in range(len(snake_list) - 5, len(snake_list)):
                segment = snake_list[i]
                print(f"   [#{i}] ({segment[0]}, {segment[1]})")
        
        print("="*60 + "\n")
    
    def direction_to_string(self, direction):
        """Convert direction tuple to readable string."""
        if direction == UP:
            return "UP ↑"
        elif direction == DOWN:
            return "DOWN ↓"
        elif direction == LEFT:
            return "LEFT ←"
        elif direction == RIGHT:
            return "RIGHT →"
        return "UNKNOWN"
    
    def convert_relative_to_absolute_direction(self, relative_action):
        """
        Convert relative action to absolute direction.
        MUST MATCH TRAINING: 0=turn right, 1=straight, 2=turn left
        
        Args:
            relative_action: 0 (turn right), 1 (straight), 2 (turn left)
            
        Returns:
            Absolute direction tuple (UP, DOWN, LEFT, RIGHT)
        """
        # Get current direction
        current_dir = self.game_engine.direction
        
        # Map relative action to new direction (MUST MATCH enhanced_dqn.py perform_action)
        # Training uses: 0=right, 1=straight, 2=left
        if current_dir == UP:
            new_dirs = [RIGHT, UP, LEFT]  # right, straight, left
        elif current_dir == RIGHT:
            new_dirs = [DOWN, RIGHT, UP]
        elif current_dir == DOWN:
            new_dirs = [LEFT, DOWN, RIGHT]
        else:  # LEFT
            new_dirs = [UP, LEFT, DOWN]
        
        return new_dirs[relative_action]
    
    def cycle_dqn_model(self):
        """Cycle to the next available DQN model."""
        if not self.available_dqn_models:
            print("No DQN models available")
            return
        
        next_index = (self.selected_model_index + 1) % len(self.available_dqn_models)
        if self.load_dqn_model(next_index):
            self.update_model_menu_option()
    
    def update_model_menu_option(self):
        """Update the menu to show the currently selected model."""
        if self.available_dqn_models:
            current_model = self.available_dqn_models[self.selected_model_index]
            model_name = current_model['filename']
            # Truncate long filenames
            if len(model_name) > 25:
                model_name = model_name[:22] + "..."
            self.menu_options[2] = f"DQN Model: {model_name}"
        else:
            self.menu_options[2] = "DQN Model: None Available"
    
    def update_available_modes(self):
        """Update the available game modes based on trained models."""
        # Start with base modes
        available_modes = [MANUAL_MODE, ASTAR_MODE, DIJKSTRA_MODE]
        
        # Check if Q-learning model is available
        q_model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
        if os.path.exists(q_model_path):
            if QLEARNING_MODE not in available_modes:
                available_modes.append(QLEARNING_MODE)
        else:
            print("Q-learning model not found. You'll need to train one first.")
            
        # Check if DQN model is available (try Enhanced first, then standard)
        enhanced_dqn_path = os.path.join(QMODEL_DIR, "snake_enhanced_dqn.pth")
        dqn_model_path = os.path.join(QMODEL_DIR, DQN_MODEL_FILE)
        if os.path.exists(enhanced_dqn_path) or os.path.exists(dqn_model_path):
            if DQN_MODE not in available_modes:
                available_modes.append(DQN_MODE)
        else:
            print("DQN model not found. You'll need to train one first.")
            
        # Update the global game modes
        global GAME_MODES
        GAME_MODES = available_modes
        
        # Reset mode index if it's out of bounds
        if self.mode_index >= len(GAME_MODES):
            self.mode_index = 0
            self.menu_options[0] = "Mode: " + GAME_MODES[self.mode_index]
    
    def start_game(self):
        """Start the game with the current selected mode and speed."""
        self.current_mode = GAME_MODES[self.mode_index]
        self.current_speed = FRAME_RATES[list(FRAME_RATES.keys())[self.speed_index]]
        self.game_state = STATE_PLAYING
        
        # Reset the game
        self.reset_game()
    
    def reset_game(self):
        """Reset the game state."""
        self.game_engine.reset_game()
        
        # If we're in a game, ensure the state is playing
        if self.game_state != STATE_MENU:
            self.game_state = STATE_PLAYING
    
    def handle_menu_input(self, key):
        """Handle input in the menu state."""
        if key == K_UP or key == K_w:
            self.selected_option = (self.selected_option - 1) % len(self.menu_options)
        elif key == K_DOWN or key == K_s:
            self.selected_option = (self.selected_option + 1) % len(self.menu_options)
        elif key == K_RETURN:
            if self.selected_option == 0:  # Mode option
                self.mode_index = (self.mode_index + 1) % len(GAME_MODES)
                self.menu_options[0] = "Mode: " + GAME_MODES[self.mode_index]
            elif self.selected_option == 1:  # Speed option
                self.speed_index = (self.speed_index + 1) % len(FRAME_RATES)
                self.menu_options[1] = "Speed: " + list(FRAME_RATES.keys())[self.speed_index]
            elif self.selected_option == 2:  # DQN Model Browser
                self.cycle_dqn_model()
            elif self.selected_option == 3:  # Start game
                self.start_game()
    
    def handle_game_input(self, key):
        """Handle input in the playing state."""
        # Only process movement keys in manual mode
        if self.current_mode == MANUAL_MODE:
            if key == K_w or key == K_UP:
                self.game_engine.change_direction(UP)
            elif key == K_s or key == K_DOWN:
                self.game_engine.change_direction(DOWN)
            elif key == K_a or key == K_LEFT:
                self.game_engine.change_direction(LEFT)
            elif key == K_d or key == K_RIGHT:
                self.game_engine.change_direction(RIGHT)
        
        # Pause game regardless of mode
        if key == K_p:
            self.game_engine.paused = not self.game_engine.paused
        
        # Toggle debug mode (G for debug)
        if key == K_g:
            self.debug_mode = not self.debug_mode
            print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            # Debug signature: 0x4761_76696E_54616E (hex)
        
        # Return to menu
        if key == K_ESCAPE:
            self.game_state = STATE_MENU
        
        # Reset game
        if key == K_r:
            self.reset_game()
    
    def handle_game_over_input(self, key):
        """Handle input in the game over state."""
        if key == K_RETURN or key == K_ESCAPE:
            self.game_state = STATE_MENU
    
    def process_events(self):
        """Process all game events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            if event.type == KEYDOWN:
                if self.game_state == STATE_MENU:
                    self.handle_menu_input(event.key)
                elif self.game_state == STATE_PLAYING:
                    self.handle_game_input(event.key)
                elif self.game_state == STATE_GAME_OVER:
                    self.handle_game_over_input(event.key)
    
    def update(self):
        """Update game logic."""
        if self.game_state == STATE_PLAYING and not self.game_engine.paused:
            # Update based on game mode
            if self.current_mode == MANUAL_MODE:
                # Manual mode - player controls the snake
                pass  # Direction is set by keyboard input
            elif self.current_mode == ASTAR_MODE:
                # A* algorithm mode
                direction = self.algorithms.get_next_move_astar()
                self.game_engine.set_direction_from_algorithm(direction)
            elif self.current_mode == DIJKSTRA_MODE:
                # Dijkstra's algorithm mode
                direction = self.algorithms.get_next_move_dijkstra()
                self.game_engine.set_direction_from_algorithm(direction)
            elif self.current_mode == QLEARNING_MODE and self.q_agent is not None:
                # Q-learning algorithm mode
                direction = self.q_agent.get_next_move_qlearning()
                self.game_engine.set_direction_from_algorithm(direction)
            elif self.current_mode == DQN_MODE and self.dqn_agent is not None:
                # Advanced DQN algorithm mode
                # Use agent's own get_state() which returns correct feature count (34 for Enhanced, 11 for old)
                state = self.dqn_agent.get_state()
                
                # Get Q-values for debugging
                with torch.no_grad():
                    q_values = self.dqn_agent.policy_net(state.unsqueeze(0))
                    self.last_q_values = q_values.squeeze().cpu().tolist()
                
                action_idx = self.dqn_agent.select_action(state, training=False)
                self.last_action = action_idx
                
                # Store state summary for debugging
                self.last_state_summary = {
                    'danger_straight': state[0].item(),
                    'danger_right': state[1].item(),
                    'danger_left': state[2].item(),
                    'food_up': state[3].item(),
                    'food_down': state[4].item(),
                    'food_left': state[5].item(),
                    'food_right': state[6].item(),
                }
                
                # Convert relative action to absolute direction
                # TRAINING MAPPING: 0=turn right, 1=straight, 2=turn left
                direction = self.convert_relative_to_absolute_direction(action_idx)
                self.game_engine.set_direction_from_algorithm(direction)
            
            # Move snake
            self.game_engine.move_snake()
            
            # Check game over
            if self.game_engine.game_over:
                # Print final game state to terminal
                self.print_final_game_state()
                
                # Check for new highscore
                self.save_highscore(self.game_engine.score)
                self.game_state = STATE_GAME_OVER
    
    def render(self):
        """Render the game."""
        # Clear screen
        self.screen.fill(BLACK)
        
        if self.game_state == STATE_MENU:
            self.render_menu()
        elif self.game_state == STATE_PLAYING:
            self.render_game()
        elif self.game_state == STATE_GAME_OVER:
            self.render_game_over()
        
        # Update the display
        pygame.display.flip()
    
    def render_menu(self):
        """Render the menu screen."""
        # Draw title
        font_title = pygame.font.Font(None, 64)
        title_text = font_title.render(GAME_TITLE, True, GREEN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Draw menu options
        font_menu = pygame.font.Font(None, 36)
        for i, option in enumerate(self.menu_options):
            if i == self.selected_option:
                color = YELLOW  # Highlight selected option
            else:
                color = WHITE
            
            text = font_menu.render(option, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 250 + i * 50))
            self.screen.blit(text, rect)
        
        # Draw instructions
        font_instructions = pygame.font.Font(None, 24)
        instructions = [
            "Use UP/DOWN or W/S keys to navigate",
            "Press ENTER to select an option",
            "In game: WASD or Arrow keys to move",
            "P to pause, R to reset, ESC to return to menu"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font_instructions.render(instruction, True, GRAY)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 400 + i * 30))
            self.screen.blit(text, rect)
            
        # Display model status (Q-Learning and DQN)
        status_font = pygame.font.Font(None, 20)
        
        # Q-learning model status
        q_model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
        if os.path.exists(q_model_path):
            q_status_text = "Q-Learning model: Trained"
            q_status_color = LIGHT_GRAY
        else:
            q_status_text = "Q-Learning model: Not trained"
            q_status_color = DARK_GRAY
        
        q_text = status_font.render(q_status_text, True, q_status_color)
        q_rect = q_text.get_rect(center=(SCREEN_WIDTH // 2, 520))
        self.screen.blit(q_text, q_rect)
        
        # DQN model status (check both Enhanced and standard)
        enhanced_dqn_path = os.path.join(QMODEL_DIR, "snake_enhanced_dqn.pth")
        dqn_model_path = os.path.join(QMODEL_DIR, DQN_MODEL_FILE)
        if os.path.exists(enhanced_dqn_path):
            dqn_status_text = "Enhanced DQN model: Trained ✓"
            dqn_status_color = GREEN
        elif os.path.exists(dqn_model_path):
            dqn_status_text = "Advanced DQN model: Trained"
            dqn_status_color = LIGHT_GRAY
        else:
            dqn_status_text = "DQN model: Not trained"
            dqn_status_color = DARK_GRAY
        
        dqn_text = status_font.render(dqn_status_text, True, dqn_status_color)
        dqn_rect = dqn_text.get_rect(center=(SCREEN_WIDTH // 2, 545))
        self.screen.blit(dqn_text, dqn_rect)
        
        # Show warning message if selected mode requires training
        if self.mode_index < len(GAME_MODES):
            warning_font = pygame.font.Font(None, 22)
            warning_text = None
            
            if GAME_MODES[self.mode_index] == QLEARNING_MODE and not os.path.exists(q_model_path):
                warning_text = "Please select 'Train Q-Learning Model' first!"
            elif GAME_MODES[self.mode_index] == DQN_MODE:
                # Check for both Enhanced and standard DQN models
                enhanced_dqn_path = os.path.join(QMODEL_DIR, "snake_enhanced_dqn.pth")
                if not (os.path.exists(dqn_model_path) or os.path.exists(enhanced_dqn_path)):
                    warning_text = "Please select 'Train Advanced DQN Model' first!"
            
            if warning_text:
                warning = warning_font.render(warning_text, True, RED)
                warning_rect = warning.get_rect(center=(SCREEN_WIDTH // 2, 570))
                self.screen.blit(warning, warning_rect)
    
    def render_game(self):
        """Render the game screen."""
        # Render game elements (snake, food, etc.)
        self.game_engine.render(self.screen)
        
        # Render game info
        font = pygame.font.Font(None, 24)
        
        # Mode
        mode_text = font.render(f"Mode: {self.current_mode}", True, WHITE)
        self.screen.blit(mode_text, (10, 10))
        
        # Speed
        speed_name = list(FRAME_RATES.keys())[self.speed_index]
        speed_text = font.render(f"Speed: {speed_name}", True, WHITE)
        self.screen.blit(speed_text, (10, 40))
        
        # Score
        score_text = font.render(f"Score: {self.game_engine.score}", True, WHITE)
        self.screen.blit(score_text, (10, 70))
        
        # Highscore
        highscore_color = YELLOW if self.game_engine.score >= self.highscore else WHITE
        highscore_text = font.render(f"Highscore: {self.highscore}", True, highscore_color)
        self.screen.blit(highscore_text, (10, 100))
        
        # Render debug overlay if debug mode is active and DQN mode
        if self.debug_mode and self.current_mode == DQN_MODE and self.dqn_agent is not None:
            debug_y_offset = 140
            
            # Q-values display
            if self.last_q_values is not None:
                q_text = font.render("Q-Values:", True, YELLOW)
                self.screen.blit(q_text, (10, debug_y_offset))
                debug_y_offset += 25
                
                actions = ["Straight", "Right", "Left"]
                for i, (action_name, q_val) in enumerate(zip(actions, self.last_q_values)):
                    color = GREEN if i == self.last_action else WHITE
                    marker = " <--" if i == self.last_action else ""
                    q_line = font.render(f"  {action_name}: {q_val:.3f}{marker}", True, color)
                    self.screen.blit(q_line, (10, debug_y_offset))
                    debug_y_offset += 20
            
            # State summary display
            if self.last_state_summary is not None:
                debug_y_offset += 5
                state_text = font.render("Danger:", True, YELLOW)
                self.screen.blit(state_text, (10, debug_y_offset))
                debug_y_offset += 25
                
                danger_items = [
                    ("Straight", self.last_state_summary.get('danger_straight', 0)),
                    ("Right", self.last_state_summary.get('danger_right', 0)),
                    ("Left", self.last_state_summary.get('danger_left', 0))
                ]
                
                for name, value in danger_items:
                    color = RED if value > 0.5 else GREEN
                    status = "YES" if value > 0.5 else "No"
                    danger_line = font.render(f"  {name}: {status}", True, color)
                    self.screen.blit(danger_line, (10, debug_y_offset))
                    debug_y_offset += 20
                
                debug_y_offset += 5
                food_text = font.render("Food Direction:", True, YELLOW)
                self.screen.blit(food_text, (10, debug_y_offset))
                debug_y_offset += 25
                
                food_items = [
                    ("Up", self.last_state_summary.get('food_up', 0)),
                    ("Down", self.last_state_summary.get('food_down', 0)),
                    ("Left", self.last_state_summary.get('food_left', 0)),
                    ("Right", self.last_state_summary.get('food_right', 0))
                ]
                
                for name, value in food_items:
                    color = GREEN if value > 0.5 else WHITE
                    status = "YES" if value > 0.5 else "No"
                    food_line = font.render(f"  {name}: {status}", True, color)
                    self.screen.blit(food_line, (10, debug_y_offset))
                    debug_y_offset += 20
            
            # Debug mode indicator
            debug_indicator = font.render("DEBUG MODE (Press G to toggle)", True, YELLOW)
            self.screen.blit(debug_indicator, (SCREEN_WIDTH - 320, 10))
        
        # Render pause message if game is paused
        if self.game_engine.paused:
            font_pause = pygame.font.Font(None, 48)
            pause_text = font_pause.render("PAUSED", True, YELLOW)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(pause_text, pause_rect)
    
    def render_game_over(self):
        """Render the game over screen."""
        # First render the final game state
        self.game_engine.render(self.screen)
        
        # Create a semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Game over message
        font_large = pygame.font.Font(None, 72)
        game_over_text = font_large.render("GAME OVER", True, RED)
        game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Score
        font_medium = pygame.font.Font(None, 48)
        score_text = font_medium.render(f"Score: {self.game_engine.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        self.screen.blit(score_text, score_rect)
        
        # Highscore
        is_new_highscore = self.game_engine.score >= self.highscore
        highscore_color = GOLD if is_new_highscore else WHITE
        highscore_text = font_medium.render(f"Highscore: {self.highscore}", True, highscore_color)
        highscore_rect = highscore_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70))
        self.screen.blit(highscore_text, highscore_rect)
        
        # New highscore message
        if is_new_highscore and self.game_engine.score > 0:
            font_new = pygame.font.Font(None, 36)
            new_text = font_new.render("🎉 NEW HIGHSCORE! 🎉", True, GOLD)
            new_rect = new_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120))
            self.screen.blit(new_text, new_rect)
        
        # Continue message
        font_small = pygame.font.Font(None, 36)
        continue_text = font_small.render("Press ENTER or ESC to continue", True, YELLOW)
        continue_rect = continue_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 160))
        self.screen.blit(continue_text, continue_rect)
    
    def run(self):
        """Start the game loop."""
        while self.running:
            # Process events
            self.process_events()
            
            # Update game logic
            self.update()
            
            # Render game
            self.render()
            
            # Control frame rate
            if self.game_state == STATE_PLAYING:
                fps = self.current_speed
            else:
                fps = 30  # Menu and game over screens run at fixed FPS
                
            self.clock.tick(fps)
        
        # Clean up
        pygame.quit()
        sys.exit()

# Run the game if this script is executed directly
if __name__ == "__main__":
    game = SnakeGame()
    game.run()