"""
UI module for the Snake Game.
Implements menu, game UI, settings, and other interface elements using Tkinter.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pygame
from constants import *

class GameUI:
    def __init__(self, root, start_game_callback, reset_game_callback):
        """Initialize the UI with callbacks for starting and resetting the game."""
        self.root = root
        self.start_game_callback = start_game_callback
        self.reset_game_callback = reset_game_callback
        
        # Configure the window
        self.root.title(GAME_TITLE)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Variables for settings
        self.selected_mode = tk.StringVar(value=MANUAL_MODE)
        self.selected_speed = tk.StringVar(value=DEFAULT_SPEED)
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for different screens
        self.menu_frame = None
        self.game_info_frame = None
        
        # Initialize menu
        self._create_menu()

    def _create_menu(self):
        """Create the main menu screen."""
        # Create a new frame for the menu
        if self.menu_frame:
            self.menu_frame.destroy()
        
        self.menu_frame = ttk.Frame(self.main_frame, padding=20)
        self.menu_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(self.menu_frame, text=GAME_TITLE, font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(self.menu_frame, text="Game Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for mode in GAME_MODES:
            ttk.Radiobutton(mode_frame, text=mode, value=mode, variable=self.selected_mode).pack(anchor=tk.W, padx=10, pady=5)
        
        # Speed selection
        speed_frame = ttk.LabelFrame(self.menu_frame, text="Game Speed")
        speed_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for speed in FRAME_RATES.keys():
            ttk.Radiobutton(speed_frame, text=speed, value=speed, variable=self.selected_speed).pack(anchor=tk.W, padx=10, pady=5)
        
        # Start button
        start_button = ttk.Button(self.menu_frame, text="Start Game", command=self._start_game)
        start_button.pack(pady=20)
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "• Manual Mode: Use WASD keys to control the snake\n"
            "• Algorithm Modes: Watch the snake find food automatically\n"
            "• Collect food to grow and increase your score\n"
            "• Avoid colliding with the snake's body"
        )
        
        instruction_label = ttk.Label(self.menu_frame, text=instructions, justify=tk.LEFT)
        instruction_label.pack(pady=10)
    
    def _create_game_info_panel(self):
        """Create the game information panel shown during gameplay."""
        if self.game_info_frame:
            self.game_info_frame.destroy()
        
        self.game_info_frame = ttk.Frame(self.root, width=INFO_WIDTH)
        self.game_info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Game info
        info_frame = ttk.LabelFrame(self.game_info_frame, text="Game Information")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Mode
        mode_label = ttk.Label(info_frame, text=f"Mode: {self.selected_mode.get()}")
        mode_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Speed
        speed_label = ttk.Label(info_frame, text=f"Speed: {self.selected_speed.get()}")
        speed_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Score
        self.score_var = tk.StringVar(value="Score: 0")
        score_label = ttk.Label(info_frame, textvariable=self.score_var)
        score_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Controls
        control_frame = ttk.LabelFrame(self.game_info_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        controls_text = (
            "W: Move Up\n"
            "A: Move Left\n"
            "S: Move Down\n"
            "D: Move Right\n"
            "P: Pause Game"
        )
        
        controls_label = ttk.Label(control_frame, text=controls_text, justify=tk.LEFT)
        controls_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.game_info_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        # Reset button
        reset_button = ttk.Button(button_frame, text="Reset Game", command=self._reset_game)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Menu button
        menu_button = ttk.Button(button_frame, text="Back to Menu", command=self._back_to_menu)
        menu_button.pack(side=tk.RIGHT, padx=5)
    
    def _start_game(self):
        """Start the game with the selected settings."""
        # Hide the menu frame
        if self.menu_frame:
            self.menu_frame.pack_forget()
        
        # Create the game info panel
        self._create_game_info_panel()
        
        # Call the start game callback with the selected settings
        mode = self.selected_mode.get()
        speed = FRAME_RATES[self.selected_speed.get()]
        self.start_game_callback(mode, speed)
    
    def _reset_game(self):
        """Reset the current game."""
        self.reset_game_callback()
    
    def _back_to_menu(self):
        """Return to the main menu."""
        # Hide the game info frame
        if self.game_info_frame:
            self.game_info_frame.pack_forget()
        
        # Create the menu
        self._create_menu()
        
        # Call the start game callback with None to indicate return to menu
        self.start_game_callback(None, None)
    
    def update_score(self, score):
        """Update the displayed score."""
        if hasattr(self, 'score_var'):
            self.score_var.set(f"Score: {score}")
    
    def show_game_over(self, score):
        """Show the game over message."""
        messagebox.showinfo("Game Over", f"Game Over! Your final score is {score}.")
        self._back_to_menu()
    
    def _on_close(self):
        """Handle window close event."""
        self.root.quit()

class GameDisplay:
    def __init__(self):
        """Initialize the pygame display for the game."""
        # Initialize pygame display
        pygame.init()
        self.screen = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        
    def get_surface(self):
        """Get the pygame surface."""
        return self.screen
    
    def update(self):
        """Update the pygame display."""
        pygame.display.flip()
    
    def quit(self):
        """Quit pygame."""
        pygame.quit()