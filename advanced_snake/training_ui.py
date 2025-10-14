"""
Enhanced Training UI for Snake Game DQN training.

This module provides a GUI interface for the headless training module with:
1. Parameter configuration (episodes, save intervals, etc.)
2. Model selection and visualization of training progress
3. Overview of existing models and their statistics
"""

import os
import sys
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import subprocess
import threading
import glob
import re
from datetime import datetime
import pandas as pd
import numpy as np
import psutil

# Training system metadata - Generated Attribution Variable Info Notation: Tan
__author__ = "Gavin Tan"
__version__ = "3.1"

# Import constants
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import *

class TrainingUI:
    def __init__(self, root):
        """Initialize the Training UI."""
        try:
            self.root = root
            self.root.title("Snake Game - DQN Training Control Center")
            
            # ENHANCED: Use constants for window sizing with better defaults
            default_width = max(TRAINING_SCREEN_WIDTH, 1400)  # Increased from 1300
            default_height = max(TRAINING_SCREEN_HEIGHT, 900)  # Increased from 800
            self.root.geometry(f"{default_width}x{default_height}")
            
            # Set reasonable minimum size (not larger than default!)
            min_width = max(MIN_TRAINING_SCREEN_WIDTH, 1200)  # Increased from 800
            min_height = max(MIN_TRAINING_SCREEN_HEIGHT, 700)  # Reasonable minimum
            self.root.minsize(min_width, min_height)
            
            # Center window on screen
            self.center_window(default_width, default_height)
            
            # Set up styles
            self.setup_styles()
            
            # Set default values
            self.model_dir = QMODEL_DIR
            self.training_process = None
            self.is_training = False
            self.models_info = []
            self.selected_model = tk.StringVar()
            self.training_log = []
            self.curriculum_advancements = []  # Track when curriculum advances: [(episode, old_stage, new_stage), ...]
            
            # Performance optimization: Track last update time and plot objects
            self.last_full_redraw = 0
            self.plot_objects = {}  # Store plot line objects for incremental updates
            self.last_episode_count = 0  # Track if new data arrived
            
            # Create main frames
        except Exception as e:
            import traceback
            print(f"Initialization error: {e}")
            print(traceback.format_exc())
        self.create_main_layout()
        
        # Initialize UI elements
        self.create_parameter_panel()
        self.create_models_panel()
        self.create_training_controls()
        self.create_statistics_panel()
        
        # Initial model scan
        self.scan_models()
        
        # Setup periodic UI updates
        self.root.after(500, self.update_ui)
    
    def center_window(self, width, height):
        """Center the window on the screen."""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set geometry
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        """Set up ttk styles for the UI."""
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#3498db")
        style.configure("TLabel", padding=5)
        style.configure("TFrame", background="#f5f5f5")
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Bold.TLabel", font=("Arial", 10, "bold"))
        style.configure("Training.TButton", background="#27ae60", foreground="black", font=("Arial", 10, "bold"))
        style.configure("Stop.TButton", background="#c0392b", foreground="black", font=("Arial", 10, "bold"))
        
        # Configure the Treeview
        style.configure("Treeview", 
                        background="#f5f5f5",
                        foreground="black",
                        rowheight=25,
                        fieldbackground="#f5f5f5")
        style.map('Treeview', background=[('selected', '#3498db')])

    def create_main_layout(self):
        """Create the main layout frames."""
        # Main container frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top-level notebook with tabs
        self.main_notebook = ttk.Notebook(self.main_frame)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Training Controls and Model Management
        self.controls_tab = ttk.Frame(self.main_notebook, padding="10")
        self.main_notebook.add(self.controls_tab, text="Training Controls")
        
        # Tab 2: Training Performance Graphs
        self.graph_tab = ttk.Frame(self.main_notebook, padding="10")
        self.main_notebook.add(self.graph_tab, text="Training Performance")
        
        # Tab 3: Model Visualization (for future enhancements)
        self.visualization_tab = ttk.Frame(self.main_notebook, padding="10")
        self.main_notebook.add(self.visualization_tab, text="Model Visualization")
        
        # Set up the Training Controls tab layout
        self.setup_controls_tab()
        
        # Set up the visualization tab
        self.setup_visualization_tab()
        
        # The graph_frame will be the graph_tab now
        self.graph_frame = self.graph_tab
        
    def setup_controls_tab(self):
        """Set up the training controls tab layout."""
        # Top frame for parameters and training controls
        self.top_frame = ttk.Frame(self.controls_tab)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left panel - for parameters
        self.param_frame = ttk.LabelFrame(self.top_frame, text="Training Parameters", padding="10")
        self.param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel - for training controls
        self.control_frame = ttk.LabelFrame(self.top_frame, text="Training Controls", padding="10")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Middle panel - for model selection
        self.middle_frame = ttk.Frame(self.controls_tab)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Models panel
        self.models_frame = ttk.LabelFrame(self.middle_frame, text="Available Models", padding="10")
        self.models_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for statistics
        self.stats_frame = ttk.LabelFrame(self.middle_frame, text="Training Statistics", padding="10")
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def setup_visualization_tab(self):
        """Set up the visualization tab with advanced model analysis features."""
        # Create a notebook for different visualization types
        self.viz_notebook = ttk.Notebook(self.visualization_tab)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Neural Network Architecture
        self.arch_tab = ttk.Frame(self.viz_notebook, padding="10")
        self.viz_notebook.add(self.arch_tab, text="Network Architecture")
        self.setup_architecture_viz()
        
        # Tab 2: Feature Importance
        self.feature_tab = ttk.Frame(self.viz_notebook, padding="10")
        self.viz_notebook.add(self.feature_tab, text="Feature Importance")
        self.setup_feature_importance_viz()
        
        # Tab 3: Live Game State Analysis
        self.live_tab = ttk.Frame(self.viz_notebook, padding="10")
        self.viz_notebook.add(self.live_tab, text="Live State Analysis")
        self.setup_live_state_viz()
    
    def setup_architecture_viz(self):
        """Set up the neural network architecture visualization."""
        # Control panel at top
        control_frame = ttk.LabelFrame(self.arch_tab, text="Model Selection", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model type selector
        ttk.Label(control_frame, text="Model Type:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.arch_model_type = tk.StringVar(value="Enhanced DQN (34 features)")
        arch_combo = ttk.Combobox(
            control_frame,
            textvariable=self.arch_model_type,
            values=["Q-Learning (Tabular)", "Enhanced DQN (34 features)"],
            state="readonly",
            width=35
        )
        arch_combo.pack(side=tk.LEFT, padx=5)
        
        # Visualize button
        viz_btn = ttk.Button(control_frame, text="Visualize Architecture", 
                            command=self.draw_architecture)
        viz_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for drawing - render engine v2.0 (GT_2025)
        canvas_frame = ttk.Frame(self.arch_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for architecture
        self.arch_fig, self.arch_ax = plt.subplots(figsize=(12, 8))
        self.arch_canvas = FigureCanvasTkAgg(self.arch_fig, master=canvas_frame)
        self.arch_canvas.draw()
        self.arch_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial draw
        self.draw_architecture()
    
    def draw_architecture(self):
        """Draw the neural network architecture diagram."""
        self.arch_ax.clear()
        self.arch_ax.axis('off')
        
        model_type = self.arch_model_type.get()
        is_qlearning = "Q-Learning" in model_type
        is_enhanced = "Enhanced" in model_type
        
        if is_qlearning:
            # Q-Learning tabular approach - visualize differently
            self.draw_qlearning_diagram()
            return
        
        if is_enhanced:
            # Enhanced DQN architecture: 34 -> 256 -> 128 -> 4 (with dueling)
            layers = [
                ("Input", 34, "State Features"),
                ("Hidden 1", 256, "ReLU"),
                ("Hidden 2", 128, "ReLU"),
                ("Value Stream", 128, "Advantage"),
                ("Output", 4, "Q-Values")
            ]
            title = "Enhanced DQN Architecture (34 features)\nWith Dueling Network, Double DQN, Prioritized Experience Replay"
        else:
            # Default to Enhanced DQN
            layers = [
                ("Input", 34, "State Features"),
                ("Hidden 1", 256, "ReLU"),
                ("Hidden 2", 128, "ReLU"),
                ("Value Stream", 128, "Advantage"),
                ("Output", 4, "Q-Values")
            ]
            title = "Enhanced DQN Architecture (34 features)\nWith Dueling Network, Double DQN, Prioritized Experience Replay"
        
        # Calculate positions
        n_layers = len(layers)
        x_positions = np.linspace(0.1, 0.9, n_layers)
        y_center = 0.5
        
        # Draw layers
        for i, (name, size, activation) in enumerate(layers):
            x = x_positions[i]
            
            # Draw rectangle for layer
            width = 0.12
            height = 0.15
            
            # Color coding
            if "Input" in name:
                color = '#3498db'  # Blue
            elif "Output" in name:
                color = '#e74c3c'  # Red
            elif "Value" in name:
                color = '#f39c12'  # Orange
            else:
                color = '#2ecc71'  # Green
            
            # Draw layer box
            rect = plt.Rectangle((x - width/2, y_center - height/2), width, height,
                                facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
            self.arch_ax.add_patch(rect)
            
            # Add layer name
            self.arch_ax.text(x, y_center + height/2 + 0.05, name,
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add layer size
            self.arch_ax.text(x, y_center, f"{size} nodes",
                            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            
            # Add activation
            self.arch_ax.text(x, y_center - height/2 - 0.05, activation,
                            ha='center', va='top', fontsize=9, style='italic')
            
            # Draw connections to next layer
            if i < n_layers - 1:
                x_next = x_positions[i + 1]
                # Draw arrow
                self.arch_ax.annotate('', xy=(x_next - width/2, y_center),
                                    xytext=(x + width/2, y_center),
                                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
                
                # Calculate parameters for this connection
                params = size * layers[i + 1][1]
                if params > 1000000:
                    param_text = f"{params/1000000:.2f}M"
                elif params > 1000:
                    param_text = f"{params/1000:.1f}K"
                else:
                    param_text = str(params)
                
                # Add parameter count on arrow
                mid_x = (x + x_next) / 2
                self.arch_ax.text(mid_x, y_center + 0.08, f"{param_text} params",
                                ha='center', va='bottom', fontsize=8, color='gray')
        
        # Add title
        self.arch_ax.text(0.5, 0.95, title,
                        ha='center', va='top', fontsize=14, fontweight='bold',
                        transform=self.arch_ax.transAxes)
        
        # Add feature details for enhanced model
        if is_enhanced:
            feature_text = "34 Features: Danger (13) | Food (6) | Navigation (12) | A* Hints (3)"
        else:
            feature_text = "11 Features: Danger (8) | Food (2) | Current Direction (1)"
        
        self.arch_ax.text(0.5, 0.02, feature_text,
                        ha='center', va='bottom', fontsize=10,
                        transform=self.arch_ax.transAxes, style='italic',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set limits
        self.arch_ax.set_xlim(0, 1)
        self.arch_ax.set_ylim(0, 1)
        
        self.arch_canvas.draw()
    
    def draw_qlearning_diagram(self):
        """Draw Q-Learning tabular approach diagram."""
        # Title
        title = "Q-Learning Architecture (Tabular Method)\nDirect State-Action Value Table"
        self.arch_ax.text(0.5, 0.95, title,
                        ha='center', va='top', fontsize=14, fontweight='bold',
                        transform=self.arch_ax.transAxes)
        
        # State representation box
        state_box = plt.Rectangle((0.05, 0.6), 0.25, 0.25,
                                  facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.7)
        self.arch_ax.add_patch(state_box)
        self.arch_ax.text(0.175, 0.725, "STATE\n(Tuple)", ha='center', va='center',
                         fontsize=12, fontweight='bold', color='white')
        self.arch_ax.text(0.175, 0.57, "11 binary features", ha='center', va='top',
                         fontsize=9, style='italic')
        
        # Q-Table box (main component)
        qtable_box = plt.Rectangle((0.37, 0.4), 0.26, 0.45,
                                   facecolor='#2ecc71', edgecolor='black', linewidth=3, alpha=0.7)
        self.arch_ax.add_patch(qtable_box)
        self.arch_ax.text(0.5, 0.75, "Q-TABLE", ha='center', va='center',
                         fontsize=14, fontweight='bold', color='white')
        self.arch_ax.text(0.5, 0.68, "Dictionary:\nState -> Actions", ha='center', va='center',
                         fontsize=10, color='white')
        
        # Show sample Q-table entry
        sample_text = "Example:\nState(1,0,1,...) ->\n{UP: 2.5, DOWN: -1.0,\n LEFT: 0.8, RIGHT: 3.2}"
        self.arch_ax.text(0.5, 0.5, sample_text, ha='center', va='center',
                         fontsize=8, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Action selection box
        action_box = plt.Rectangle((0.70, 0.6), 0.25, 0.25,
                                   facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.7)
        self.arch_ax.add_patch(action_box)
        self.arch_ax.text(0.825, 0.725, "ACTION\n(Absolute)", ha='center', va='center',
                         fontsize=12, fontweight='bold', color='white')
        self.arch_ax.text(0.825, 0.57, "UP/DOWN/LEFT/RIGHT", ha='center', va='top',
                         fontsize=9, style='italic')
        
        # Arrows
        # State -> Q-Table
        self.arch_ax.annotate('', xy=(0.37, 0.725), xytext=(0.30, 0.725),
                             arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        self.arch_ax.text(0.335, 0.75, 'Lookup', ha='center', fontsize=9)
        
        # Q-Table -> Action
        self.arch_ax.annotate('', xy=(0.70, 0.725), xytext=(0.63, 0.725),
                             arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        self.arch_ax.text(0.665, 0.75, 'Max Q', ha='center', fontsize=9)
        
        # Update rule box
        update_box = plt.Rectangle((0.15, 0.15), 0.70, 0.18,
                                   facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.7)
        self.arch_ax.add_patch(update_box)
        self.arch_ax.text(0.5, 0.29, "Q-Learning Update Rule", ha='center', va='top',
                         fontsize=11, fontweight='bold')
        
        update_formula = r"$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$"
        self.arch_ax.text(0.5, 0.21, update_formula, ha='center', va='center',
                         fontsize=12, family='serif')
        
        # Advantages box
        adv_text = ("+ Perfect memory of visited states\n"
                   "+ No neural network complexity\n"
                   "+ Fast training (~1000 episodes)\n"
                   "+ Deterministic action selection\n"
                   "+ Easy to interpret and debug")
        self.arch_ax.text(0.05, 0.05, adv_text, ha='left', va='bottom',
                         fontsize=9, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Set limits
        self.arch_ax.set_xlim(0, 1)
        self.arch_ax.set_ylim(0, 1)
        
        self.arch_canvas.draw()
    
    def setup_feature_importance_viz(self):
        """Set up the feature importance visualization."""
        # Control panel
        control_frame = ttk.LabelFrame(self.feature_tab, text="Analysis Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selector
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.feat_model_path = tk.StringVar(value="No model selected")
        model_entry = ttk.Entry(control_frame, textvariable=self.feat_model_path, width=40)
        model_entry.pack(side=tk.LEFT, padx=5)
        
        # Browse button
        browse_btn = ttk.Button(control_frame, text="Browse", 
                               command=self.browse_model_for_features)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        analyze_btn = ttk.Button(control_frame, text="Analyze Importance",
                                command=self.analyze_feature_importance,
                                style="Training.TButton")
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Sample size
        ttk.Label(control_frame, text="Samples:").pack(side=tk.LEFT, padx=(10, 5))
        self.feat_samples = tk.StringVar(value="1000")
        sample_spin = ttk.Spinbox(control_frame, from_=100, to=10000, increment=100,
                                 textvariable=self.feat_samples, width=10)
        sample_spin.pack(side=tk.LEFT)
        
        # Canvas for plotting
        canvas_frame = ttk.Frame(self.feature_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.feat_fig, self.feat_ax = plt.subplots(figsize=(12, 10))
        self.feat_canvas = FigureCanvasTkAgg(self.feat_fig, master=canvas_frame)
        self.feat_canvas.draw()
        self.feat_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Show placeholder
        self.show_feature_placeholder()
    
    def show_feature_placeholder(self):
        """Show placeholder for feature importance."""
        self.feat_ax.clear()
        self.feat_ax.text(0.5, 0.5, "Select a model and click 'Analyze Importance'\nto view feature rankings",
                         ha='center', va='center', fontsize=14, transform=self.feat_ax.transAxes)
        self.feat_ax.axis('off')
        self.feat_canvas.draw()
    
    def browse_model_for_features(self):
        """Browse for a model file to analyze."""
        filepath = filedialog.askopenfilename(
            initialdir=self.model_dir,
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if filepath:
            self.feat_model_path.set(filepath)
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance using gradient-based analysis."""
        model_path = self.feat_model_path.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        try:
            # Determine model type from filename
            is_enhanced = "enhanced" in model_path.lower()
            
            if is_enhanced:
                feature_names = self.get_enhanced_feature_names()
                num_features = 34
            else:
                # Assume Q-Learning model (doesn't use feature analysis)
                self.add_to_log("Feature analysis is only available for Enhanced DQN models.", log_type="system")
                return
            
            n_samples = int(self.feat_samples.get())
            
            self.add_to_log(f"Loading model from {os.path.basename(model_path)}...", log_type="system")
            
            # Load the model - system ID: 0x47_54_32_30_32_35
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Import the appropriate model class
            if is_enhanced:
                from enhanced_dqn import EnhancedDQNAgent
                from game_engine import GameEngine
                
                # Create a temporary game engine and agent
                game = GameEngine()
                agent = EnhancedDQNAgent(game)
                
                # Load the model weights
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.policy_net.load_state_dict(checkpoint)
                
                agent.policy_net.eval()
                model = agent.policy_net
            else:
                from enhanced_dqn import EnhancedDQNAgent
                from game_engine import GameEngine
                
                # Create a temporary game engine and agent
                game = GameEngine()
                agent = EnhancedDQNAgent(game)
                
                # Load the model weights
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.policy_net.load_state_dict(checkpoint)
                
                agent.policy_net.eval()
                model = agent.policy_net
            
            self.add_to_log(f"Model loaded successfully. Analyzing {num_features} features using {n_samples} samples...", log_type="system")
            
            # Calculate gradient-based feature importance
            importance_scores = self.calculate_gradient_importance(model, num_features, n_samples, device)
            
            self.add_to_log(f"Gradient-based importance analysis completed.", log_type="system")
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1]
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_scores = importance_scores[sorted_indices]
            
            # Normalize scores to 0-1 range for better visualization
            if sorted_scores.max() > 0:
                normalized_scores = sorted_scores / sorted_scores.max()
            else:
                normalized_scores = sorted_scores
            
            # Plot
            self.feat_ax.clear()
            
            # Create horizontal bar chart
            y_pos = np.arange(len(sorted_names))
            colors = plt.cm.RdYlGn(normalized_scores)
            
            bars = self.feat_ax.barh(y_pos, sorted_scores, color=colors, alpha=0.8, edgecolor='black')
            
            self.feat_ax.set_yticks(y_pos)
            self.feat_ax.set_yticklabels(sorted_names, fontsize=8)
            self.feat_ax.set_xlabel('Importance Score (Avg Gradient Magnitude)', fontsize=12, fontweight='bold')
            self.feat_ax.set_title(f'Feature Importance Ranking (Gradient-Based Analysis)\n({num_features} features, {n_samples} samples analyzed)',
                                  fontsize=14, fontweight='bold')
            self.feat_ax.grid(axis='x', alpha=0.3)
            
            # Invert y-axis so most important is on top
            self.feat_ax.invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                self.feat_ax.text(score + sorted_scores.max() * 0.01, bar.get_y() + bar.get_height()/2,
                                f'{score:.4f}', va='center', fontsize=7)
            
            # Add category legend
            if is_enhanced:
                legend_text = "Categories: Danger (red) | Food (blue) | Navigation (green) | A* Hints (orange)"
            else:
                legend_text = "Categories: Danger (red) | Food (blue) | Direction (green)"
            
            self.feat_ax.text(0.98, 0.02, legend_text,
                            transform=self.feat_ax.transAxes,
                            ha='right', va='bottom', fontsize=9, style='italic',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Add method explanation
            method_text = "Method: Gradient-based importance - measures average gradient magnitude\nfor each feature across sampled states"
            self.feat_ax.text(0.02, 0.02, method_text,
                            transform=self.feat_ax.transAxes,
                            ha='left', va='bottom', fontsize=8, style='italic',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            self.feat_canvas.draw()
            
            self.add_to_log("Feature importance analysis completed.", log_type="system")
            
            # Log top 5 features with their scores
            top_5_text = ", ".join([f"{sorted_names[i]} ({sorted_scores[i]:.4f})" for i in range(min(5, len(sorted_names)))])
            self.add_to_log(f"Top 5 features: {top_5_text}", log_type="system")
            
            # Log bottom 5 features (least important)
            if len(sorted_names) >= 10:
                bottom_5_text = ", ".join([f"{sorted_names[i]} ({sorted_scores[i]:.4f})" for i in range(len(sorted_names)-5, len(sorted_names))])
                self.add_to_log(f"Bottom 5 features: {bottom_5_text}", log_type="system")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze features: {str(e)}")
            self.add_to_log(f"Error analyzing features: {str(e)}", log_type="system")
            import traceback
            traceback.print_exc()
    
    def calculate_gradient_importance(self, model, num_features, n_samples, device):
        """
        Calculate feature importance using gradient-based analysis.
        
        This method measures how much each input feature affects the model's output
        by calculating the gradient of the Q-values with respect to each input feature.
        
        Args:
            model: The neural network model
            num_features: Number of input features
            n_samples: Number of random samples to analyze
            device: torch device (CPU or CUDA)
        
        Returns:
            numpy array of importance scores for each feature
        """
        model.eval()
        
        # Storage for gradients
        feature_gradients = np.zeros(num_features)
        
        # Generate random states
        # We'll use states sampled from a reasonable distribution
        # Features are typically in range [0, 1] for normalized states
        states = np.random.rand(n_samples, num_features).astype(np.float32)
        
        # Also add some states with more structure (not purely random)
        # Mix of zero, half, and full activation patterns
        structured_samples = n_samples // 4
        if structured_samples > 0:
            states[:structured_samples] = np.random.choice([0.0, 0.5, 1.0], 
                                                          size=(structured_samples, num_features))
        
        self.add_to_log(f"Calculating gradients for {n_samples} sampled states...", log_type="system")
        
        # Calculate gradients for each sample
        for i in range(n_samples):
            # Convert to tensor
            state_tensor = torch.FloatTensor(states[i:i+1]).to(device)
            state_tensor.requires_grad = True
            
            # Forward pass
            with torch.set_grad_enabled(True):
                q_values = model(state_tensor)
                
                # Calculate gradient of max Q-value w.r.t. input
                max_q = q_values.max()
                max_q.backward()
            
            # Get gradients
            if state_tensor.grad is not None:
                grad = state_tensor.grad.cpu().numpy()[0]
                # Use absolute gradient as importance measure
                feature_gradients += np.abs(grad)
            
            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                progress = (i + 1) / n_samples * 100
                self.add_to_log(f"Progress: {progress:.1f}% ({i+1}/{n_samples} samples)", log_type="system")
        
        # Average the gradients
        feature_gradients /= n_samples
        
        self.add_to_log(f"Gradient calculation completed. Computing importance scores...", log_type="system")
        
        return feature_gradients
    
    def get_enhanced_feature_names(self):
        """Get feature names for enhanced DQN (34 features)."""
        return [
            # Danger detection (13 features)
            "Danger Straight", "Danger Right", "Danger Left",
            "Extended Danger Straight", "Extended Danger Right", "Extended Danger Left",
            "Trap Ahead", "Body Proximity Front", "Body Proximity Right", "Body Proximity Left",
            "Wall Distance Up", "Wall Distance Down", "Wall Distance Left", "Wall Distance Right",
            
            # Food direction (6 features)
            "Food Up", "Food Down", "Food Left", "Food Right",
            "Food Distance X", "Food Distance Y",
            
            # Navigation (12 features)
            "Current Direction Up", "Current Direction Down", 
            "Current Direction Left", "Current Direction Right",
            "Available Space Ahead", "Available Space Right", "Available Space Left",
            "Snake Length", "Tail Direction X", "Tail Direction Y",
            "Distance to Tail", "Moves Until Tail Clear",
            
            # A* hints (3 features)
            "A* Suggests Straight", "A* Suggests Right", "A* Suggests Left"
        ]
    
    def setup_live_state_viz(self):
        """Set up the live game state analysis visualization."""
        # Info label
        info_frame = ttk.Frame(self.live_tab)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, 
                 text="Live State Analysis - Load a model and click 'Analyze Current State' during gameplay",
                 font=("Arial", 10, "italic")).pack()
        
        # Control panel
        control_frame = ttk.LabelFrame(self.live_tab, text="Analysis Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selector
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.live_model_path = tk.StringVar(value="No model selected")
        model_entry = ttk.Entry(control_frame, textvariable=self.live_model_path, width=40)
        model_entry.pack(side=tk.LEFT, padx=5)
        
        browse_btn = ttk.Button(control_frame, text="Browse",
                               command=self.browse_model_for_live)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        analyze_btn = ttk.Button(control_frame, text="Analyze Random State",
                                command=self.analyze_live_state,
                                style="Training.TButton")
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area with split view
        content_frame = ttk.Frame(self.live_tab)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Feature values
        left_frame = ttk.LabelFrame(content_frame, text="Current State Features", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create scrollable text widget for features
        feat_scroll = ttk.Scrollbar(left_frame)
        feat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.live_features_text = tk.Text(left_frame, wrap=tk.WORD, 
                                         yscrollcommand=feat_scroll.set, height=20, width=40)
        self.live_features_text.pack(fill=tk.BOTH, expand=True)
        feat_scroll.config(command=self.live_features_text.yview)
        
        # Right panel - Q-values and visualizations
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Q-values display
        q_frame = ttk.LabelFrame(right_frame, text="Q-Values & Recommendations", padding="10")
        q_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for Q-values and radar chart
        self.live_fig, self.live_axes = plt.subplots(2, 1, figsize=(8, 10))
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=q_frame)
        self.live_canvas.draw()
        self.live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Show placeholder
        self.show_live_placeholder()
    
    def show_live_placeholder(self):
        """Show placeholder for live state analysis."""
        for ax in self.live_axes:
            ax.clear()
            ax.axis('off')
        
        self.live_axes[0].text(0.5, 0.5, "Select a model and click\n'Analyze Random State'",
                              ha='center', va='center', fontsize=12,
                              transform=self.live_axes[0].transAxes)
        
        self.live_canvas.draw()
    
    def browse_model_for_live(self):
        """Browse for a model file for live analysis."""
        filepath = filedialog.askopenfilename(
            initialdir=self.model_dir,
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if filepath:
            self.live_model_path.set(filepath)
    
    def analyze_live_state(self):
        """Analyze and display current game state with real model predictions."""
        model_path = self.live_model_path.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        try:
            # Determine model type
            is_enhanced = "enhanced" in model_path.lower()
            
            if is_enhanced:
                feature_names = self.get_enhanced_feature_names()
                num_features = 34
            else:
                # Q-Learning models don't use DQN features
                self.add_to_log("Live analysis is only available for Enhanced DQN models.", log_type="system")
                return
            
            self.add_to_log(f"Loading model for live analysis...", log_type="system")
            
            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Import the appropriate model class
            if is_enhanced:
                from enhanced_dqn import EnhancedDQNAgent
                from game_engine import GameEngine
                
                # Create a temporary game engine and agent
                game = GameEngine()
                agent = EnhancedDQNAgent(game)
                
                # Load the model weights
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.policy_net.load_state_dict(checkpoint)
                
                agent.policy_net.eval()
                
                # Generate a real game state - simulate forward a few steps for variety
                game.reset_game()
                
                # Randomly simulate 0-20 steps to get more varied game states
                import random
                num_steps = random.randint(0, 20)
                self.add_to_log(f"Simulating {num_steps} steps to create varied game state...", log_type="system")
                
                for step in range(num_steps):
                    state = agent.get_state()
                    if isinstance(state, list):
                        state_tensor_temp = torch.FloatTensor(state).unsqueeze(0).to(device)
                    else:
                        state_tensor_temp = state.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        q_values_temp = agent.policy_net(state_tensor_temp)
                        action = torch.argmax(q_values_temp).item()
                    
                    # Convert relative action to absolute direction
                    # Actions: 0=straight, 1=right, 2=left
                    current_dir_tuple = game.direction
                    direction_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
                    direction_reverse = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
                    
                    current_dir_num = direction_map.get(current_dir_tuple, 3)
                    
                    # Map relative to absolute direction
                    if action == 0:  # Straight
                        new_direction = current_dir_num
                    elif action == 1:  # Right
                        new_direction = (current_dir_num + 1) % 4
                    elif action == 2:  # Left
                        new_direction = (current_dir_num - 1) % 4
                    else:
                        new_direction = current_dir_num
                    
                    # Set the new direction in the game
                    game.change_direction(direction_reverse[new_direction])
                    
                    # Move the game forward
                    game.move_snake()
                    
                    if game.game_over:
                        # Game ended, reset and continue
                        game.reset_game()
                        break
                
                # Now get the current state after simulation
                state_values_raw = agent.get_state()
                
                # Convert to numpy array for display and ensure on CPU
                if isinstance(state_values_raw, torch.Tensor):
                    state_values = state_values_raw.cpu().numpy()
                    state_tensor = state_values_raw.unsqueeze(0).to(device)
                else:
                    state_values = np.array(state_values_raw)
                    state_tensor = torch.FloatTensor(state_values).unsqueeze(0).to(device)
                    
                with torch.no_grad():
                    q_values_tensor = agent.policy_net(state_tensor)
                    q_values = q_values_tensor.cpu().numpy()[0]
                
                # Get A* suggestion
                head = game.snake[0]
                food = game.food
                path = agent.algorithms._find_path_astar(head, food)
                
                if path and len(path) > 1:
                    # Determine A* suggested direction
                    next_pos = path[1]
                    
                    # Convert direction tuple to numeric (UP=0, DOWN=1, LEFT=2, RIGHT=3)
                    direction_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
                    current_direction = direction_map.get(game.direction, 3)  # Default to RIGHT
                    
                    # Calculate relative direction
                    dx = next_pos[0] - head[0]
                    dy = next_pos[1] - head[1]
                    
                    # Map to relative action (0=straight, 1=right, 2=left)
                    if current_direction == 0:  # UP
                        if dy == -1: astar_action = 0  # straight
                        elif dx == 1: astar_action = 1  # right
                        elif dx == -1: astar_action = 2  # left
                        else: astar_action = 0
                    elif current_direction == 1:  # DOWN
                        if dy == 1: astar_action = 0  # straight
                        elif dx == -1: astar_action = 1  # right
                        elif dx == 1: astar_action = 2  # left
                        else: astar_action = 0
                    elif current_direction == 2:  # LEFT
                        if dx == -1: astar_action = 0  # straight
                        elif dy == -1: astar_action = 1  # right
                        elif dy == 1: astar_action = 2  # left
                        else: astar_action = 0
                    else:  # RIGHT
                        if dx == 1: astar_action = 0  # straight
                        elif dy == 1: astar_action = 1  # right
                        elif dy == -1: astar_action = 2  # left
                        else: astar_action = 0
                    
                    astar_action_name = ["Straight", "Right", "Left"][astar_action]
                else:
                    astar_action = None
                    astar_action_name = "No path found"
                
            else:
                from enhanced_dqn import EnhancedDQNAgent
                from game_engine import GameEngine
                
                # Create a temporary game engine and agent
                game = GameEngine()
                agent = EnhancedDQNAgent(game)
                
                # Load the model weights
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.policy_net.load_state_dict(checkpoint)
                
                agent.policy_net.eval()
                
                # Generate a real game state
                game.reset_game()
                # Generate a real game state
                game.reset_game()
                state_values_raw = agent.get_state()
                
                # Convert to numpy array for display and ensure on CPU
                if isinstance(state_values_raw, torch.Tensor):
                    state_values = state_values_raw.cpu().numpy()
                    state_tensor = state_values_raw.unsqueeze(0).to(device)
                else:
                    state_values = np.array(state_values_raw)
                    state_tensor = torch.FloatTensor(state_values).unsqueeze(0).to(device)
                    
                with torch.no_grad():
                    q_values_tensor = agent.policy_net(state_tensor)
                    q_values = q_values_tensor.cpu().numpy()[0]
                
                astar_action = None
                astar_action_name = "N/A"
            
            # Enhanced DQN uses 3 actions (Straight, Right, Left)
            # Standard DQN might use 4 actions (including Opposite)
            if len(q_values) == 3:
                action_names = ["Straight", "Right", "Left"]
            else:
                action_names = ["Straight", "Right", "Left", "Opposite"]
            
            best_action = np.argmax(q_values)
            
            self.add_to_log(f"State generated: Snake length {len(game.snake)}, Score {game.score}", log_type="system")
            
            # Update feature text
            self.live_features_text.config(state=tk.NORMAL)
            self.live_features_text.delete(1.0, tk.END)
            
            self.live_features_text.insert(tk.END, "=" * 40 + "\n", "header")
            self.live_features_text.insert(tk.END, "CURRENT STATE FEATURES\n", "header")
            self.live_features_text.insert(tk.END, "=" * 40 + "\n", "header")
            self.live_features_text.insert(tk.END, f"Game State: Length={len(game.snake)}, Score={game.score}\n\n", "info")
            
            # Group features by category
            if is_enhanced:
                categories = [
                    ("DANGER DETECTION", 0, 13),
                    ("FOOD INFORMATION", 13, 19),
                    ("NAVIGATION", 19, 31),
                    ("A* HINTS", 31, 34)
                ]
            else:
                categories = [
                    ("DANGER", 0, 8),
                    ("FOOD", 8, 10),
                    ("DIRECTION", 10, 11)
                ]
            
            for cat_name, start, end in categories:
                self.live_features_text.insert(tk.END, f"\n{cat_name}:\n", "category")
                self.live_features_text.insert(tk.END, "-" * 40 + "\n", "separator")
                
                for i in range(start, min(end, len(feature_names))):
                    value = state_values[i]
                    bar = "#" * int(value * 20)
                    self.live_features_text.insert(tk.END, 
                        f"{feature_names[i]:30s} {value:5.3f} {bar}\n")
            
            # Add recommendations
            self.live_features_text.insert(tk.END, f"\n{'='*40}\n", "header")
            self.live_features_text.insert(tk.END, "RECOMMENDATIONS\n", "header")
            self.live_features_text.insert(tk.END, f"{'='*40}\n", "header")
            self.live_features_text.insert(tk.END, f"\nDQN Best Action: {action_names[best_action]}\n", "bold")
            self.live_features_text.insert(tk.END, f"Q-Value: {q_values[best_action]:.2f}\n", "info")
            self.live_features_text.insert(tk.END, f"All Q-Values: {[f'{q:.2f}' for q in q_values]}\n\n", "info")
            
            if is_enhanced and astar_action is not None:
                self.live_features_text.insert(tk.END, f"A* Suggests: {astar_action_name}\n", "bold")
                if best_action == astar_action or (best_action == 0 and astar_action == 0):
                    self.live_features_text.insert(tk.END, "[YES] DQN agrees with A*!\n", "success")
                else:
                    self.live_features_text.insert(tk.END, "[NO] DQN disagrees with A*\n", "warning")
                    self.live_features_text.insert(tk.END, f"  A* path length: {len(path) if path else 0} steps\n", "info")
            elif is_enhanced:
                self.live_features_text.insert(tk.END, f"A* Status: {astar_action_name}\n", "warning")
            
            self.live_features_text.config(state=tk.DISABLED)
            
            # Configure tags for styling
            self.live_features_text.tag_config("header", font=("Courier", 10, "bold"))
            self.live_features_text.tag_config("category", font=("Courier", 9, "bold"), foreground="blue")
            self.live_features_text.tag_config("separator", foreground="gray")
            self.live_features_text.tag_config("bold", font=("Courier", 10, "bold"))
            self.live_features_text.tag_config("info", foreground="darkblue")
            self.live_features_text.tag_config("success", foreground="green", font=("Courier", 10, "bold"))
            self.live_features_text.tag_config("warning", foreground="orange", font=("Courier", 10, "bold"))
            
            # Plot Q-values
            self.live_axes[0].clear()
            colors = ['green' if i == best_action else 'skyblue' for i in range(len(q_values))]
            bars = self.live_axes[0].bar(action_names, q_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            self.live_axes[0].set_ylabel('Q-Value', fontsize=12, fontweight='bold')
            self.live_axes[0].set_title('Action Q-Values (Real Model Predictions)', fontsize=14, fontweight='bold')
            self.live_axes[0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, qval in zip(bars, q_values):
                height = bar.get_height()
                self.live_axes[0].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{qval:.2f}',
                                      ha='center', va='bottom' if height >= 0 else 'top', 
                                      fontweight='bold', fontsize=10)
            
            # Highlight A* suggestion if enhanced
            if is_enhanced and astar_action is not None:
                # Draw vertical line at A* suggested action
                self.live_axes[0].axvline(x=astar_action, color='red', linestyle='--', 
                                         linewidth=2.5, alpha=0.7, label=f'A* suggests: {astar_action_name}')
                self.live_axes[0].legend(fontsize=10)
            
            # Add best action annotation
            self.live_axes[0].annotate(f'Best: {action_names[best_action]}', 
                                      xy=(best_action, q_values[best_action]),
                                      xytext=(best_action, q_values[best_action] + abs(q_values.max() - q_values.min()) * 0.15),
                                      arrowprops=dict(arrowstyle='->', color='green', lw=2),
                                      fontsize=11, fontweight='bold', color='green',
                                      ha='center')
            
            # Plot feature radar chart (top features)
            self.live_axes[1].clear()
            
            # Select top 8 features by value (excluding very small values)
            significant_indices = [i for i, val in enumerate(state_values) if val > 0.01]
            if len(significant_indices) > 8:
                top_indices = sorted(significant_indices, key=lambda i: state_values[i], reverse=True)[:8]
            else:
                # If fewer than 8 significant features, just take top 8 overall
                top_indices = np.argsort(state_values)[-8:][::-1]
            
            top_names = [feature_names[i][:25] for i in top_indices]  # Truncate names
            top_values = state_values[top_indices]
            
            # Normalize for radar chart (0-1 range)
            if top_values.max() > 0:
                top_values_normalized = top_values / max(top_values.max(), 1.0)
            else:
                top_values_normalized = top_values
            
            # Radar chart
            angles = np.linspace(0, 2 * np.pi, len(top_names), endpoint=False).tolist()
            top_values_plot = top_values_normalized.tolist()
            
            # Close the plot
            angles += angles[:1]
            top_values_plot += top_values_plot[:1]
            
            # Remove the old axis and create a new polar axis in the same position
            self.live_axes[1].remove()
            self.live_axes[1] = self.live_fig.add_subplot(2, 1, 2, projection='polar')
            self.live_axes[1].plot(angles, top_values_plot, 'o-', linewidth=2.5, color='blue', markersize=8)
            self.live_axes[1].fill(angles, top_values_plot, alpha=0.25, color='blue')
            self.live_axes[1].set_xticks(angles[:-1])
            self.live_axes[1].set_xticklabels(top_names, size=8)
            self.live_axes[1].set_ylim(0, 1)
            self.live_axes[1].set_title('Top 8 Active Features (Radar Chart)', 
                                       fontsize=12, fontweight='bold', pad=20)
            self.live_axes[1].grid(True, alpha=0.3)
            
            # Add value labels on the radar chart
            for angle, value, name in zip(angles[:-1], top_values_plot[:-1], top_names):
                self.live_axes[1].text(angle, value + 0.1, f'{value:.2f}', 
                                      ha='center', va='center', fontsize=7,
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            self.live_canvas.draw()
            
            self.add_to_log(f"Analyzed game state: {num_features} features, Best action: {action_names[best_action]} (Q={q_values[best_action]:.2f})", 
                          log_type="system")
            
            if is_enhanced and astar_action is not None:
                agreement = "agrees" if best_action == astar_action else "disagrees"
                self.add_to_log(f"DQN {agreement} with A* suggestion ({astar_action_name})", log_type="system")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze state: {str(e)}")
            self.add_to_log(f"Error analyzing state: {str(e)}", log_type="system")
            import traceback
            traceback.print_exc()

    def create_parameter_panel(self):
        """Create the training parameters panel."""
        # Episodes
        episodes_frame = ttk.Frame(self.param_frame)
        episodes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(episodes_frame, text="Episodes:").pack(side=tk.LEFT)
        
        self.episodes_var = tk.StringVar(value="1000")
        self.episodes_combo = ttk.Combobox(
            episodes_frame, 
            textvariable=self.episodes_var, 
            values=["100", "500", "1000", "2000", "5000", "10000"],
            width=10
        )
        self.episodes_combo.pack(side=tk.RIGHT)
        
        # Model type selection - NEW!
        model_type_frame = ttk.Frame(self.param_frame)
        model_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_type_frame, text="Model Type:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        self.model_type_var = tk.StringVar(value="Enhanced DQN (34 features)")
        self.model_type_combo = ttk.Combobox(
            model_type_frame,
            textvariable=self.model_type_var,
            values=["Q-Learning (Tabular)", "Enhanced DQN (34 features)"],
            width=35,
            state="readonly"
        )
        self.model_type_combo.pack(side=tk.RIGHT)
        self.model_type_combo.bind("<<ComboboxSelected>>", self.on_model_type_changed)
        
        # Add info label for model type
        model_info_frame = ttk.Frame(self.param_frame)
        model_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_info_label = ttk.Label(
            model_info_frame,
            text="Enhanced: A* guidance, curriculum learning, trap detection",
            foreground="green",
            font=('Arial', 8, 'italic')
        )
        self.model_info_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Save interval
        save_frame = ttk.Frame(self.param_frame)
        save_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(save_frame, text="Save Interval:").pack(side=tk.LEFT)
        
        self.save_interval_var = tk.StringVar(value="50")
        self.save_interval_combo = ttk.Combobox(
            save_frame, 
            textvariable=self.save_interval_var,
            values=["10", "25", "50", "100", "200"],
            width=10
        )
        self.save_interval_combo.pack(side=tk.RIGHT)
        
        # Batch size
        batch_frame = ttk.Frame(self.param_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        
        self.batch_size_var = tk.StringVar(value="64")
        self.batch_size_combo = ttk.Combobox(
            batch_frame,
            textvariable=self.batch_size_var,
            values=["32", "64", "128", "256", "512"],
            width=10
        )
        self.batch_size_combo.pack(side=tk.RIGHT)
        
        # Learning rate with spinbox
        lr_frame = ttk.Frame(self.param_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        
        # Container for spinbox controls
        lr_control_frame = ttk.Frame(lr_frame)
        lr_control_frame.pack(side=tk.RIGHT)
        
        self.learning_rate_var = tk.DoubleVar(value=0.002)
        
        # Spinbox for learning rate with increment/decrement of 0.001
        self.learning_rate_spinbox = ttk.Spinbox(
            lr_control_frame,
            from_=0.0001,
            to=0.01,
            increment=0.001,
            textvariable=self.learning_rate_var,
            width=12,
            format="%.4f"
        )
        self.learning_rate_spinbox.pack(side=tk.LEFT)
        
        # Start from checkpoint
        checkpoint_frame = ttk.Frame(self.param_frame)
        checkpoint_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(checkpoint_frame, text="Start from New Model:").pack(side=tk.LEFT)
        
        # Check if models exist first to set appropriate default
        model_exists = os.path.exists(os.path.join(QMODEL_DIR, DQN_MODEL_FILE))
        
        # Now inverted: checked = new model, unchecked = continue training
        self.use_checkpoint_var = tk.BooleanVar(value=False)  # Default to continue
        self.use_checkpoint_check = ttk.Checkbutton(
            checkpoint_frame,
            variable=self.use_checkpoint_var,
            text="(Unchecked = continue training)",
            command=self.update_model_number_hint  # Update hint when checkbox changes
        )
        self.use_checkpoint_check.pack(side=tk.RIGHT)
        
        # Model Number (for creating numbered models)
        model_num_frame = ttk.Frame(self.param_frame)
        model_num_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_num_frame, text="Model Number:").pack(side=tk.LEFT)
        
        # Auto-find next available number
        self.next_model_number = self.find_next_model_number()
        
        self.model_number_var = tk.StringVar(value="")
        model_num_entry = ttk.Entry(model_num_frame, textvariable=self.model_number_var, width=10)
        model_num_entry.pack(side=tk.LEFT, padx=5)
        
        # Dynamic label that updates based on selection
        self.model_number_hint_label = ttk.Label(
            model_num_frame, 
            text=f"(Leave empty for default, next available: {self.next_model_number})", 
            font=('Arial', 8, 'italic'),
            foreground='gray'
        )
        self.model_number_hint_label.pack(side=tk.LEFT)
        
        # Add trace to update hint when model number changes
        self.model_number_var.trace('w', lambda *args: self.update_model_number_hint())
        
        # Model directory
        dir_frame = ttk.Frame(self.param_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="Models Directory:").pack(side=tk.LEFT)
        
        self.dir_entry = ttk.Entry(dir_frame, width=25)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.dir_entry.insert(0, os.path.abspath(self.model_dir))
        
        self.browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_model_dir)
        self.browse_btn.pack(side=tk.RIGHT)
        
        # ============================================================
        # STUCK DETECTION CONTROLS - NEW!
        # ============================================================
        separator = ttk.Separator(self.param_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        stuck_header_frame = ttk.Frame(self.param_frame)
        stuck_header_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(stuck_header_frame, text="Stuck Detection & Epsilon Boost", 
                 font=('Arial', 10, 'bold'), foreground='darkblue').pack(side=tk.LEFT)
        
        # Enable/Disable stuck detection
        stuck_enable_frame = ttk.Frame(self.param_frame)
        stuck_enable_frame.pack(fill=tk.X, pady=5)
        
        self.stuck_detection_var = tk.BooleanVar(value=ENABLE_STUCK_DETECTION)
        stuck_check = ttk.Checkbutton(
            stuck_enable_frame,
            text="Enable Stuck Detection",
            variable=self.stuck_detection_var,
            command=self.on_stuck_detection_toggled
        )
        stuck_check.pack(side=tk.LEFT)
        
        ttk.Label(stuck_enable_frame, text="(Boosts epsilon when agent is stuck)", 
                 font=('Arial', 8, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Create a frame for stuck detection parameters (can be disabled)
        self.stuck_params_frame = ttk.Frame(self.param_frame)
        self.stuck_params_frame.pack(fill=tk.X, pady=5)
        
        # Sensitivity (Stuck Counter Threshold)
        sensitivity_frame = ttk.Frame(self.stuck_params_frame)
        sensitivity_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(sensitivity_frame, text="Sensitivity:").pack(side=tk.LEFT)
        
        self.stuck_sensitivity_var = tk.IntVar(value=STUCK_COUNTER_THRESHOLD)
        sensitivity_scale = ttk.Scale(
            sensitivity_frame,
            from_=1,
            to=10,
            variable=self.stuck_sensitivity_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self.stuck_sensitivity_label.config(
                text=f"{int(float(v))} checks ({int(float(v)) * 50} episodes)"
            )
        )
        sensitivity_scale.pack(side=tk.LEFT, padx=5)
        
        self.stuck_sensitivity_label = ttk.Label(
            sensitivity_frame,
            text=f"{STUCK_COUNTER_THRESHOLD} checks ({STUCK_COUNTER_THRESHOLD * 50} episodes)"
        )
        self.stuck_sensitivity_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sensitivity_frame, text="(1=aggressive, 10=conservative)", 
                 font=('Arial', 7, 'italic'), foreground='gray').pack(side=tk.LEFT)
        
        # Cooldown Period
        cooldown_frame = ttk.Frame(self.stuck_params_frame)
        cooldown_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(cooldown_frame, text="Cooldown:").pack(side=tk.LEFT)
        
        self.stuck_cooldown_var = tk.IntVar(value=STUCK_BOOST_COOLDOWN)
        cooldown_scale = ttk.Scale(
            cooldown_frame,
            from_=50,
            to=500,
            variable=self.stuck_cooldown_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self.stuck_cooldown_label.config(text=f"{int(float(v))} episodes")
        )
        cooldown_scale.pack(side=tk.LEFT, padx=5)
        
        self.stuck_cooldown_label = ttk.Label(
            cooldown_frame,
            text=f"{STUCK_BOOST_COOLDOWN} episodes"
        )
        self.stuck_cooldown_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(cooldown_frame, text="(time between boosts)", 
                 font=('Arial', 7, 'italic'), foreground='gray').pack(side=tk.LEFT)
        
        # Boost Amount
        boost_frame = ttk.Frame(self.stuck_params_frame)
        boost_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(boost_frame, text="Boost Amount:").pack(side=tk.LEFT)
        
        self.stuck_boost_var = tk.DoubleVar(value=STUCK_EPSILON_BOOST)
        boost_scale = ttk.Scale(
            boost_frame,
            from_=0.05,
            to=0.30,
            variable=self.stuck_boost_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self.stuck_boost_label.config(text=f"+{float(v):.2f}")
        )
        boost_scale.pack(side=tk.LEFT, padx=5)
        
        self.stuck_boost_label = ttk.Label(
            boost_frame,
            text=f"+{STUCK_EPSILON_BOOST:.2f}"
        )
        self.stuck_boost_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(boost_frame, text="(epsilon increase)", 
                 font=('Arial', 7, 'italic'), foreground='gray').pack(side=tk.LEFT)
        
        # Improvement Threshold
        improvement_frame = ttk.Frame(self.stuck_params_frame)
        improvement_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(improvement_frame, text="Min Improvement:").pack(side=tk.LEFT)
        
        self.stuck_improvement_var = tk.DoubleVar(value=STUCK_IMPROVEMENT_THRESHOLD)
        improvement_scale = ttk.Scale(
            improvement_frame,
            from_=2.0,
            to=15.0,
            variable=self.stuck_improvement_var,
            orient=tk.HORIZONTAL,
            length=150,
             command=lambda v: self.stuck_improvement_label.config(text=f"{float(v):.1f} points")
        )
        improvement_scale.pack(side=tk.LEFT, padx=5)
        
        self.stuck_improvement_label = ttk.Label(
            improvement_frame,
            text=f"{STUCK_IMPROVEMENT_THRESHOLD:.1f} points"
        )
        self.stuck_improvement_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(improvement_frame, text="(to avoid stuck)", 
                 font=('Arial', 7, 'italic'), foreground='gray').pack(side=tk.LEFT)
        
        # Set initial state of stuck params
        self.on_stuck_detection_toggled()

    def create_models_panel(self):
        """Create the available models panel with a Treeview."""
        # ENHANCED: Better layout with guaranteed button visibility
        
        # Create a frame for the Treeview and scrollbar
        tree_frame = ttk.Frame(self.models_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create the Treeview with better height configuration
        self.models_tree = ttk.Treeview(
            tree_frame,
            columns=("Model", "Episodes", "Best Score", "Last Updated"),
            show="headings",
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set,
            height=8  # ENHANCED: Fixed height to ensure buttons are visible
        )
        
        # Configure the scrollbars
        y_scrollbar.config(command=self.models_tree.yview)
        x_scrollbar.config(command=self.models_tree.xview)
        
        # Set up the columns with better widths
        self.models_tree.heading("Model", text="Model")
        self.models_tree.heading("Episodes", text="Episodes")
        self.models_tree.heading("Best Score", text="Best Score")
        self.models_tree.heading("Last Updated", text="Last Updated")
        
        self.models_tree.column("Model", width=250, minwidth=150)  # ENHANCED: Better widths
        self.models_tree.column("Episodes", width=100, minwidth=80, anchor=tk.CENTER)
        self.models_tree.column("Best Score", width=100, minwidth=80, anchor=tk.CENTER)
        self.models_tree.column("Last Updated", width=180, minwidth=120, anchor=tk.CENTER)
        
        self.models_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add a selection event
        self.models_tree.bind("<<TreeviewSelect>>", self.on_model_select)
        
        # ENHANCED: Add double-click to load stats
        self.models_tree.bind("<Double-Button-1>", lambda e: self.load_model_stats())
        
        # ENHANCED: Actions frame with better styling and guaranteed visibility
        actions_frame = ttk.Frame(self.models_frame)
        actions_frame.pack(fill=tk.X, pady=(10, 5), padx=5)
        
        # Add a separator for visual clarity
        ttk.Separator(self.models_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))
        
        # Button container with centered layout
        button_container = ttk.Frame(actions_frame)
        button_container.pack(expand=True)
        
        # Refresh button with icon-like styling
        self.refresh_btn = ttk.Button(
            button_container, 
            text="Refresh Models", 
            command=self.scan_models,
            width=18
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Load model button
        self.load_btn = ttk.Button(
            button_container, 
            text="Load Stats", 
            command=self.load_model_stats,
            width=18
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Delete button with warning color
        self.delete_btn = ttk.Button(
            button_container, 
            text="Delete Model", 
            command=self.delete_model,
            width=18
        )
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Add info label for user guidance
        info_label = ttk.Label(
            self.models_frame,
            text="Tip: Double-click a model to load its statistics",
            font=('Arial', 8, 'italic'),
            foreground='gray'
        )
        info_label.pack(pady=(5, 0))

    def create_training_controls(self):
        """Create training control buttons."""
        # GPU/Device info
        device_frame = ttk.Frame(self.control_frame)
        device_frame.pack(fill=tk.X, pady=5)
        
        # Enhanced CUDA detection
        cuda_available = torch.cuda.is_available()
        device_type = "GPU" if cuda_available else "CPU"
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        
        device_text = f"Training Device: {device_type}\n{device_name}"
        
        # Add CUDA troubleshooting info if not available
        if not cuda_available:
            try:
                # Try to import subprocess to check if nvidia-smi is available
                import subprocess
                try:
                    result = subprocess.run(['nvidia-smi'], 
                                           capture_output=True, 
                                           text=True,
                                           timeout=5)
                    if result.returncode == 0:
                        # NVIDIA driver is installed but PyTorch can't see it
                        device_text += "\n\nGPU detected but PyTorch can't use it."
                        device_text += "\nPossible issues:"
                        device_text += "\n- PyTorch CUDA version mismatch"
                        device_text += "\n- Run check_cuda.py to fix"
                except (subprocess.SubprocessError, FileNotFoundError):
                    # No NVIDIA driver found
                    device_text += "\n\nNo NVIDIA GPU driver detected"
            except ImportError:
                pass
        else:
            # Show CUDA version and device properties if available
            cuda_version = torch.version.cuda
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)  # GB
            
            device_text += f"\nCUDA: {cuda_version}"
            device_text += f"\nCompute: {device_props.major}.{device_props.minor}"
            device_text += f"\nMemory: {total_memory:.2f} GB"
            
        self.device_label = ttk.Label(device_frame, text=device_text, style="Bold.TLabel")
        self.device_label.pack(fill=tk.X)
        
        # Add check CUDA button
        check_cuda_btn = ttk.Button(
            device_frame, 
            text="Check CUDA", 
            command=self.check_cuda_setup
        )
        check_cuda_btn.pack(pady=(5, 0))
        
        # Memory usage (will update dynamically)
        self.memory_frame = ttk.Frame(self.control_frame)
        self.memory_frame.pack(fill=tk.X, pady=5)
        
        if torch.cuda.is_available():
            memory_text = "VRAM: Calculating..."
        else:
            memory_text = "RAM: Calculating..."
        
        self.memory_label = ttk.Label(self.memory_frame, text=memory_text)
        self.memory_label.pack(fill=tk.X)
        
        # Start button
        self.start_btn = ttk.Button(
            self.control_frame, 
            text="Start Training", 
            command=self.start_training,
            style="Training.TButton"
        )
        self.start_btn.pack(fill=tk.X, pady=5)
        
        # Stop button
        self.stop_btn = ttk.Button(
            self.control_frame, 
            text="Stop Training", 
            command=self.stop_training,
            style="Stop.TButton",
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Training status
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(status_frame, text="Status:", style="Bold.TLabel").pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def create_statistics_panel(self):
        """Create the training statistics panel."""
        # Statistics notebook with tabs
        self.stats_notebook = ttk.Notebook(self.stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Model info tab
        self.model_info_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.model_info_frame, text="Model Info")
        
        # Create basic model info display
        model_info_inner = ttk.Frame(self.model_info_frame, padding=10)
        model_info_inner.pack(fill=tk.BOTH, expand=True)
        
        # Model name
        name_frame = ttk.Frame(model_info_inner)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(name_frame, text="Model Name:", width=15).pack(side=tk.LEFT)
        self.model_name_label = ttk.Label(name_frame, text="None selected")
        self.model_name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Training Progress
        progress_frame = ttk.Frame(model_info_inner)
        progress_frame.pack(fill=tk.X, pady=2)
        ttk.Label(progress_frame, text="Training Episodes:", width=15).pack(side=tk.LEFT)
        self.progress_label = ttk.Label(progress_frame, text="0")
        self.progress_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Best Score
        score_frame = ttk.Frame(model_info_inner)
        score_frame.pack(fill=tk.X, pady=2)
        ttk.Label(score_frame, text="Best Score:", width=15).pack(side=tk.LEFT)
        self.score_label = ttk.Label(score_frame, text="0")
        self.score_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Last Updated
        updated_frame = ttk.Frame(model_info_inner)
        updated_frame.pack(fill=tk.X, pady=2)
        ttk.Label(updated_frame, text="Last Updated:", width=15).pack(side=tk.LEFT)
        self.updated_label = ttk.Label(updated_frame, text="Never")
        self.updated_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Model Parameters
        params_frame = ttk.Frame(model_info_inner)
        params_frame.pack(fill=tk.X, pady=2)
        ttk.Label(params_frame, text="Parameters:", width=15).pack(side=tk.LEFT)
        self.params_label = ttk.Label(params_frame, text="Unknown")
        self.params_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Training log tab
        self.log_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.log_frame, text="Training Log")
        
        # Create training log text widget
        log_inner = ttk.Frame(self.log_frame, padding=5)
        log_inner.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        log_scroll = ttk.Scrollbar(log_inner)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_inner, wrap=tk.WORD, yscrollcommand=log_scroll.set, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Configure text appearance
        self.log_text.config(state=tk.DISABLED)
        
        # System log tab
        self.system_log_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.system_log_frame, text="System Log")
        
        # Create system log text widget
        system_log_inner = ttk.Frame(self.system_log_frame, padding=5)
        system_log_inner.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar for system log
        system_log_scroll = ttk.Scrollbar(system_log_inner)
        system_log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.system_log_text = tk.Text(system_log_inner, wrap=tk.WORD, yscrollcommand=system_log_scroll.set, height=10)
        self.system_log_text.pack(fill=tk.BOTH, expand=True)
        system_log_scroll.config(command=self.system_log_text.yview)
        
        # Configure text appearance
        self.system_log_text.config(state=tk.DISABLED)
        
        # Create the matplotlib figure for training graph
        self.setup_training_graph()

    def setup_training_graph(self):
        """Set up the matplotlib figure for the training graph."""
        # Clear any existing widgets
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Add control buttons at the top
        control_frame = ttk.Frame(self.graph_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Reset graph button
        reset_btn = ttk.Button(
            control_frame,
            text="Reset Graphs",
            command=self.reset_training_graphs
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_label = ttk.Label(
            control_frame,
            text="Tip: Use Reset Graphs when starting a new model to clear old data",
            font=("Arial", 9, "italic")
        )
        info_label.pack(side=tk.LEFT, padx=10)
            
        # Create a 2x1 grid (top-bottom layout) of useful training metrics
        self.fig, self.axs = plt.subplots(2, 1, figsize=(16, 10))
        self.fig.suptitle('Training Performance Analysis', fontsize=18, fontweight='bold')
        
        # Top: Score progression with running average
        self.axs[0].set_title('Score Progression', fontsize=14, fontweight='bold')
        self.axs[0].set_xlabel('Episode', fontsize=12)
        self.axs[0].set_ylabel('Score', fontsize=12)
        self.axs[0].grid(True, alpha=0.3)
        
        # Bottom: Epsilon decay over time
        self.axs[1].set_title('Exploration Rate (Epsilon)', fontsize=14, fontweight='bold')
        self.axs[1].set_xlabel('Episode', fontsize=12)
        self.axs[1].set_ylabel('Epsilon', fontsize=12)
        self.axs[1].grid(True, alpha=0.3)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.93)
        
        # Create canvas that fills the tab
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.graph_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def reset_training_graphs(self):
        """Reset training graphs and clear accumulated data."""
        confirm = messagebox.askyesno(
            "Reset Graphs",
            "This will clear all training data from the graphs.\n\n"
            "The model files will NOT be affected.\n\n"
            "Continue?"
        )
        
        if confirm:
            # Clear training data
            if hasattr(self, 'training_data'):
                self.training_data = {
                    'scores': [],
                    'running_avgs': [],
                    'steps': [],
                    'best_score': 0,
                    'losses': [],
                    'q_values': [],
                    'epsilon_values': [],  # Reset epsilon tracking
                    'lr_values': []  # NEW: Reset learning rate tracking
                }
            
            # Clear curriculum advancements
            self.curriculum_advancements = []
            
            # Reinitialize graphs
            self.setup_training_graph()
            
            # Add log message
            self.add_to_log("Training graphs reset. Ready for new training session.")

    def update_training_graph(self, scores=None, running_avgs=None, losses=None, q_values=None, force_full_redraw=False):
        """
        Update the training graph with useful metrics for DQN analysis.
        
        PERFORMANCE OPTIMIZED:
        - Supports incremental updates (just append new data points)
        - Only does full redraw when force_full_redraw=True or every 50 episodes
        - Reuses plot objects instead of recreating them
        - Gradient indicators are updated only on full redraws for performance
        """
        if not scores or len(scores) == 0:
            return
        
        # OPTIMIZATION: Force full redraw every 50 episodes to update gradient indicators
        # Gradient indicators show learning momentum and need periodic refresh
        current_episode = len(scores)
        force_gradient_update = (current_episode % 50 == 0)
        
        # OPTIMIZATION: Incremental update if possible
        if not force_full_redraw and not force_gradient_update and hasattr(self, 'plot_objects') and self.plot_objects:
            # Try incremental update (much faster)
            try:
                episodes = list(range(1, len(scores) + 1))
                
                # Update line data instead of redrawing
                if 'score_line' in self.plot_objects:
                    self.plot_objects['score_line'].set_data(episodes, scores)
                if 'avg_line' in self.plot_objects and running_avgs and len(running_avgs) == len(scores):
                    self.plot_objects['avg_line'].set_data(episodes, running_avgs)
                
                # Update epsilon line on second graph
                if 'epsilon_line' in self.plot_objects:
                    epsilon_values = self.training_data.get('epsilon_values', [])
                    if epsilon_values and len(epsilon_values) >= len(scores):
                        self.plot_objects['epsilon_line'].set_data(episodes, epsilon_values[:len(scores)])
                
                # Update LR line on second graph (secondary y-axis)
                if 'lr_line' in self.plot_objects:
                    lr_values = self.training_data.get('lr_values', [])
                    if lr_values and len(lr_values) >= len(scores):
                        self.plot_objects['lr_line'].set_data(episodes, lr_values[:len(scores)])
                
                # Rescale axes
                for ax in self.axs:
                    ax.relim()
                    ax.autoscale_view()
                
                # Redraw canvas (lightweight)
                self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
                return  # Skip full redraw
            except Exception as e:
                # If incremental update fails, fall back to full redraw
                print(f"Incremental update failed, doing full redraw: {e}")
                pass
        
        # FULL REDRAW (only when necessary)
        # Clear all plots
        for i in range(2):
            self.axs[i].clear()
        
        # Reset plot objects dictionary
        self.plot_objects = {}
        
        # ===== GRAPH 1: Score Progression (Left) =====
        self.axs[0].set_title('Score Progression', fontsize=14, fontweight='bold')
        self.axs[0].set_xlabel('Episode', fontsize=12)
        self.axs[0].set_ylabel('Score', fontsize=12)
        
        if scores and len(scores) > 0:
            episodes = list(range(1, len(scores) + 1))
            
            # Plot individual scores with low alpha - STORE REFERENCE
            score_line, = self.axs[0].plot(episodes, scores, 'b-', linewidth=0.8, alpha=0.3, label='Score')
            self.plot_objects['score_line'] = score_line
            
            # Plot running average - STORE REFERENCE
            if running_avgs and len(running_avgs) == len(scores):
                avg_line, = self.axs[0].plot(episodes, running_avgs, 'r-', linewidth=2.5, label='Avg (100)', zorder=10)
                self.plot_objects['avg_line'] = avg_line
            
            # Add curriculum stage thresholds (static, no need to store)
            thresholds = [20, 50, 100, 200]
            colors = ['green', 'orange', 'purple', 'red']
            labels = ['Stage 0->1', 'Stage 1->2', 'Stage 2->3', 'Stage 3->4']
            for threshold, color, label in zip(thresholds, colors, labels):
                self.axs[0].axhline(y=threshold, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label=label)
            
            # ===== OPTIMIZATION: Limit curriculum advancement annotations =====
            # Only show last 5 advancements to reduce clutter on long training runs
            if hasattr(self, 'curriculum_advancements') and self.curriculum_advancements:
                recent_advancements = self.curriculum_advancements[-5:]  # Last 5 only
                for episode, old_stage, new_stage in recent_advancements:
                    if episode <= len(scores):
                        score_at_advancement = scores[episode - 1] if episode > 0 else 0
                        
                        # Add vertical line at advancement
                        self.axs[0].axvline(x=episode, color='cyan', linestyle=':', linewidth=2, alpha=0.8, zorder=5)
                        
                        # Add star marker
                        self.axs[0].scatter([episode], [score_at_advancement], 
                                              marker='*', s=400, color='cyan', 
                                              edgecolors='darkblue', linewidths=2, zorder=15,
                                              label=f'Stage {old_stage}->{new_stage}' if episode == recent_advancements[0][0] else '')
                        
                        # OPTIMIZATION: Simplify annotations - no arrows for better performance
                        self.axs[0].text(episode, score_at_advancement + 20, f'S{new_stage}',
                                        fontsize=9, fontweight='bold', color='darkblue',
                                        ha='center', va='bottom',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7))
            
            # Add max score annotation (simplified)
            max_score = max(scores)
            max_idx = scores.index(max_score)
            self.axs[0].scatter([episodes[max_idx]], [max_score], color='gold', s=100, zorder=15, edgecolors='black', linewidths=2)
            self.axs[0].text(episodes[max_idx], max_score + 10, f'Best: {max_score:.0f}',
                            fontsize=10, fontweight='bold', ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))
            
            # ===== GRADIENT INDICATORS: Show rate of change in average scores =====
            # Calculate gradients (rate of change) for different time scales
            if running_avgs and len(running_avgs) >= 100:
                current_avg = running_avgs[-1]
                
                # Gradient 1: Current to Initial (overall learning rate)
                initial_avg = running_avgs[0] if running_avgs else 0
                total_episodes = len(running_avgs)
                gradient_initial = (current_avg - initial_avg) / total_episodes if total_episodes > 0 else 0
                
                # Gradient 2: Current to Mid-point (mid-term learning rate)
                mid_idx = len(running_avgs) // 2
                mid_avg = running_avgs[mid_idx]
                mid_episodes = len(running_avgs) - mid_idx
                gradient_mid = (current_avg - mid_avg) / mid_episodes if mid_episodes > 0 else 0
                
                # Gradient 3: Current to Last 100 (recent learning momentum)
                last_100_start_idx = max(0, len(running_avgs) - 100)
                last_100_start_avg = running_avgs[last_100_start_idx]
                last_100_episodes = len(running_avgs) - last_100_start_idx
                gradient_recent = (current_avg - last_100_start_avg) / last_100_episodes if last_100_episodes > 0 else 0
                
                # Determine colors based on gradient values (green = positive, red = negative, yellow = near zero)
                def get_gradient_color(gradient_value):
                    if gradient_value > 0.05:
                        return '#00CC00'  # Bright green - strong positive
                    elif gradient_value > 0.01:
                        return '#88FF88'  # Light green - weak positive
                    elif gradient_value > -0.01:
                        return '#FFFF00'  # Yellow - stagnant
                    elif gradient_value > -0.05:
                        return '#FFAA00'  # Orange - weak negative
                    else:
                        return '#FF0000'  # Red - strong negative
                
                # Create gradient indicator boxes in the upper right corner of the graph
                y_max = max(scores) if scores else 100
                x_max = len(scores)
                
                # Position boxes in upper right corner
                box_x = x_max * 0.75  # 75% across the x-axis
                box_y_start = y_max * 0.95  # Start at 95% of max y
                box_spacing = y_max * 0.08  # Space between boxes
                
                # Box 1: Overall gradient (current to initial)
                gradient_1_color = get_gradient_color(gradient_initial)
                self.axs[0].text(box_x, box_y_start, 
                                f'Overall: {gradient_initial:+.3f} pts/ep',
                                fontsize=9, fontweight='bold', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor=gradient_1_color, 
                                         edgecolor='black', linewidth=1.5, alpha=0.85),
                                zorder=20)
                
                # Box 2: Mid-term gradient (current to mid-point)
                gradient_2_color = get_gradient_color(gradient_mid)
                self.axs[0].text(box_x, box_y_start - box_spacing, 
                                f'Mid-term: {gradient_mid:+.3f} pts/ep',
                                fontsize=9, fontweight='bold', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor=gradient_2_color, 
                                         edgecolor='black', linewidth=1.5, alpha=0.85),
                                zorder=20)
                
                # Box 3: Recent gradient (last 100 episodes)
                gradient_3_color = get_gradient_color(gradient_recent)
                self.axs[0].text(box_x, box_y_start - 2 * box_spacing, 
                                f'Recent: {gradient_recent:+.3f} pts/ep',
                                fontsize=9, fontweight='bold', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor=gradient_3_color, 
                                         edgecolor='black', linewidth=1.5, alpha=0.85),
                                zorder=20)
                
                # Add a small legend explaining the gradient indicators
                self.axs[0].text(box_x, box_y_start - 3 * box_spacing - 5,
                                '(Rate of avg score change)',
                                fontsize=7, ha='left', va='top', style='italic',
                                color='gray', zorder=20)
            
            # OPTIMIZATION: Simplify legend - fewer columns
            self.axs[0].legend(loc='upper left', fontsize=8, ncol=1)
        
        self.axs[0].grid(True, alpha=0.3)
        
        # ===== GRAPH 2: Epsilon & Learning Rate Decay (Right) =====
        self.axs[1].set_title('Exploration Rate (Epsilon) & Learning Rate', fontsize=14, fontweight='bold')
        self.axs[1].set_xlabel('Episode', fontsize=12)
        self.axs[1].set_ylabel('Epsilon', fontsize=12, color='purple')
        self.axs[1].tick_params(axis='y', labelcolor='purple')
        
        if scores and len(scores) > 0:
            # OPTIMIZATION: Use cached epsilon/LR values from training data
            # This avoids expensive recalculation on every redraw
            if hasattr(self, 'training_data') and 'epsilon_values' in self.training_data and self.training_data['epsilon_values']:
                epsilon_values = self.training_data['epsilon_values']
            else:
                # Fallback: Calculate epsilon decay ONCE and cache it
                if not hasattr(self, '_cached_epsilon') or len(self._cached_epsilon) < len(scores):
                    epsilon_decay = 0.997
                    epsilon_start = 1.0
                    
                    epsilon_values = []
                    current_epsilon = epsilon_start
                    stage_minimums = {0: 0.10, 1: 0.05, 2: 0.04, 3: 0.02, 4: 0.01}
                    
                    for ep in range(len(scores)):
                        # Determine curriculum stage (simplified)
                        if running_avgs and len(running_avgs) > ep:
                            avg = running_avgs[ep]
                            if avg >= 200:
                                stage = 4
                            elif avg >= 100:
                                stage = 3
                            elif avg >= 50:
                                stage = 2
                            elif avg >= 20:
                                stage = 1
                            else:
                                stage = 0
                        else:
                            stage = 0
                        
                        stage_min = stage_minimums.get(stage, 0.01)
                        if current_epsilon > stage_min:
                            current_epsilon *= epsilon_decay
                        else:
                            current_epsilon = stage_min
                            
                        epsilon_values.append(current_epsilon)
                    
                    self._cached_epsilon = epsilon_values  # Cache for next time
                else:
                    epsilon_values = self._cached_epsilon
            
            # OPTIMIZATION: Use cached LR values from training data
            if hasattr(self, 'training_data') and 'lr_values' in self.training_data and self.training_data['lr_values']:
                lr_values = self.training_data['lr_values']
            else:
                # Fallback: Calculate LR decay ONCE and cache it
                if not hasattr(self, '_cached_lr') or len(self._cached_lr) < len(scores):
                    lr_values = []
                    current_lr = 0.005  # Stage 0 starting LR
                    
                    stage_lr_starts = {0: 0.005, 1: 0.003, 2: 0.002, 3: 0.001, 4: 0.0005}
                    
                    prev_stage = 0
                    for ep in range(len(scores)):
                        # Determine curriculum stage (simplified)
                        if running_avgs and len(running_avgs) > ep:
                            avg = running_avgs[ep]
                            if avg >= 200:
                                stage = 4
                            elif avg >= 100:
                                stage = 3
                            elif avg >= 50:
                                stage = 2
                            elif avg >= 20:
                                stage = 1
                            else:
                                stage = 0
                        else:
                            stage = 0
                        
                        # Reset LR when stage advances
                        if stage != prev_stage:
                            current_lr = stage_lr_starts.get(stage, 0.001)
                            prev_stage = stage
                    
                    # Use decay parameters from constants.py
                    stage_lr_min = STAGE_LR_MINIMUMS.get(stage, 0.0002)
                    lr_decay_rate = STAGE_LR_DECAY.get(stage, 0.9995)
                    
                    if current_lr > stage_lr_min:
                        current_lr *= lr_decay_rate
                    else:
                        current_lr = stage_lr_min
                    
                    lr_values.append(current_lr)
                
                    self._cached_lr = lr_values  # Cache for next time
                else:
                    lr_values = self._cached_lr
            
            # Ensure epsilon_values matches scores length
            if len(epsilon_values) > len(scores):
                epsilon_values = epsilon_values[:len(scores)]
            elif len(epsilon_values) < len(scores):
                # Pad with last value if needed
                if epsilon_values:
                    epsilon_values.extend([epsilon_values[-1]] * (len(scores) - len(epsilon_values)))
            
            # Ensure lr_values matches scores length
            if len(lr_values) > len(scores):
                lr_values = lr_values[:len(scores)]
            elif len(lr_values) < len(scores):
                if lr_values:
                    lr_values.extend([lr_values[-1]] * (len(scores) - len(lr_values)))
                else:
                    lr_values = [0.002] * len(scores)  # Default fallback
            
            episodes = list(range(1, len(epsilon_values) + 1))
            
            # Plot epsilon on primary axis - STORE REFERENCE
            epsilon_line, = self.axs[1].plot(episodes, epsilon_values, 'purple', linewidth=2.5, label='Epsilon', zorder=5)
            self.plot_objects['epsilon_line'] = epsilon_line
            
            # OPTIMIZATION: Simplify stage minimums - just show as text, no lines
            # (Lines add visual clutter and slow down rendering)
            stage_text = "Min eps: S0=0.10, S1=0.05, S2=0.04, S3=0.02, S4=0.01"
            self.axs[1].text(0.02, 0.95, stage_text, transform=self.axs[1].transAxes,
                            fontsize=7, color='gray', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # OPTIMIZATION: Detect epsilon boosts but limit markers to last 3 only
            boost_episodes = []
            if len(epsilon_values) > 1:
                for i in range(1, len(epsilon_values)):
                    # If epsilon increased by more than 0.05, it's likely a stuck detection boost
                    if epsilon_values[i] > epsilon_values[i-1] + 0.05:
                        boost_episodes.append(i+1)
            
            # Only show last 3 boosts to reduce clutter
            for boost_ep in boost_episodes[-3:]:
                if boost_ep <= len(epsilon_values):
                    self.axs[1].axvline(x=boost_ep, color='orange', linestyle='--', linewidth=1.5, alpha=0.4, zorder=3)
                    self.axs[1].scatter([boost_ep], [epsilon_values[boost_ep-1]], color='orange', s=80, zorder=10, 
                                          edgecolors='red', linewidths=1.5, marker='^')
            
            # Highlight current epsilon (simplified - no annotation)
            if len(epsilon_values) > 0:
                current_eps = epsilon_values[-1]
                self.axs[1].scatter([len(epsilon_values)], [current_eps], color='purple', s=100, zorder=10, edgecolors='black', linewidths=2)
                self.axs[1].text(len(epsilon_values) * 0.70, current_eps + 0.15,
                                f'eps={current_eps:.4f}',
                                fontsize=9, fontweight='bold', color='purple')
            
            # Create secondary y-axis for learning rate
            ax2 = self.axs[1].twinx()
            ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Plot learning rate - STORE REFERENCE
            lr_line, = ax2.plot(episodes, lr_values, color='green', linewidth=2.5, label='Learning Rate', linestyle='--', zorder=4)
            self.plot_objects['lr_line'] = lr_line
            
            # Set y-axis limits
            self.axs[1].set_ylim(-0.05, 1.05)
            ax2.set_ylim(0, max(lr_values) * 1.2 if lr_values else 0.006)
            
            # Combine legends
            lines1, labels1 = self.axs[1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        self.axs[1].grid(True, alpha=0.3)
        
        # Update the overall figure
        self.fig.suptitle('Training Performance Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, top=0.93)
        
        # OPTIMIZATION: Use draw_idle() instead of draw() for better performance
        # draw_idle() defers the actual rendering until the system is idle
        self.canvas.draw_idle()

    def browse_model_dir(self):
        """Open a dialog to select the models directory."""
        directory = filedialog.askdirectory(
            initialdir=self.model_dir,
            title="Select Models Directory"
        )
        
        if directory:
            self.model_dir = directory
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            self.scan_models()

    def scan_models(self):
        """Scan for models in the models directory and update the treeview."""
        # Clear existing entries
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        
        self.models_info = []
        
        # Get the directory from the entry field
        dir_path = self.dir_entry.get()
        
        if not os.path.isdir(dir_path):
            messagebox.showerror("Error", f"Directory not found: {dir_path}")
            return
        
        # Find all .pth files (DQN models) in the directory
        model_files = glob.glob(os.path.join(dir_path, "*.pth"))
        
        # Also find Q-Learning model (.pkl file)
        qlearning_file = os.path.join(dir_path, QMODEL_FILE)
        if os.path.exists(qlearning_file):
            model_files.append(qlearning_file)
        
        for model_path in model_files:
            filename = os.path.basename(model_path)
            
            # Extract episode number from checkpoint filenames
            episodes = 0
            match = re.search(r'ep(\d+)', filename)
            if match:
                episodes = int(match.group(1))
            
            # Get file modification time
            mod_time = os.path.getmtime(model_path)
            last_updated = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            
            # Try to load the model to get stats
            best_score = "Unknown"
            
            # For simplicity, just show the file info
            self.models_info.append({
                "filename": filename,
                "path": model_path,
                "episodes": episodes,
                "best_score": best_score,
                "last_updated": last_updated,
                "is_qlearning": filename.endswith('.pkl')
            })
            
            # Add to treeview
            self.models_tree.insert("", tk.END, values=(
                filename,
                episodes if episodes > 0 else "",
                best_score,
                last_updated
            ))

    def load_model_stats(self):
        """Load and display statistics for the selected model."""
        selected_items = self.models_tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Please select a model first.")
            return
        
        # Get the selected model
        item_id = selected_items[0]
        item_values = self.models_tree.item(item_id, "values")
        model_filename = item_values[0]
        
        # Find the model info
        model_info = next((m for m in self.models_info if m["filename"] == model_filename), None)
        if not model_info:
            messagebox.showerror("Error", "Model information not found.")
            return
        
        # BUGFIX: Extract model number from filename and populate UI field
        # This fixes the issue where selecting a model doesn't set the correct model number for training
        import re
        model_number_match = re.search(r'_(\d+)\.pth$', model_filename)
        if model_number_match:
            model_number = model_number_match.group(1)
            self.model_number_var.set(model_number)
            self.add_to_log(f"Selected model #{model_number}: {model_filename}", log_type="system")
        else:
            # No model number in filename (default model)
            self.model_number_var.set("")
            self.add_to_log(f"Selected default model: {model_filename}", log_type="system")
        
        # Update the UI with the model information
        self.model_name_label.config(text=model_filename)
        self.progress_label.config(text=str(model_info["episodes"]) if model_info["episodes"] > 0 else "Unknown")
        self.score_label.config(text=str(model_info["best_score"]))
        self.updated_label.config(text=model_info["last_updated"])
        
        # Try to load the model to get parameters - this is just a placeholder
        self.params_label.config(text="DQN with Double, Dueling, PER")
        
        # Try multiple possible history files
        base_model_name = os.path.splitext(model_filename)[0]
        
        # Check if this is a Q-Learning model
        is_qlearning = model_info.get('is_qlearning', False)
        
        if is_qlearning:
            # Q-Learning specific history file
            possible_history_paths = [
                os.path.join(os.path.dirname(model_info["path"]), "qlearning_training_stats.json"),
            ]
        else:
            # DQN history files
            possible_history_paths = [
                # Specific history file for this model
                os.path.join(os.path.dirname(model_info["path"]), f"{base_model_name}_history.json"),
                # Generic history file in the same directory
                os.path.join(os.path.dirname(model_info["path"]), "snake_dqn_model_history.json"),
                # Interrupted training file
                os.path.join(os.path.dirname(model_info["path"]), f"{base_model_name}_interrupted_history.json"),
            ]
        
        scores = []
        running_avgs = []
        losses = []
        q_values = []
        history_found = False
        
        # Try each possible history file
        for history_path in possible_history_paths:
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                        
                        if is_qlearning:
                            # Q-Learning uses different field names
                            scores = history.get("scores", [])
                            running_avgs = history.get("running_avg", [])  # Note: different key
                            # Q-Learning doesn't have losses/q_values in the same format
                            losses = []
                            q_values = []
                        else:
                            # DQN format
                            scores = history.get("scores", [])
                            running_avgs = history.get("running_avgs", [])
                            losses = history.get("losses", [])
                            q_values = history.get("q_values", [])
                        
                        # Update best score from history if available
                        if scores:
                            best_from_history = max(scores)
                            self.score_label.config(text=str(int(best_from_history)))
                        
                        # Update episodes from history if available  
                        if scores:
                            self.progress_label.config(text=str(len(scores)))
                    
                    # Log the successful loading
                    model_type = "Q-Learning" if is_qlearning else "DQN"
                    self.add_to_log(f"Loaded {model_type} training history from {os.path.basename(history_path)}", log_type="system")
                    history_found = True
                    break
                except Exception as e:
                    self.add_to_log(f"Error loading history from {os.path.basename(history_path)}: {str(e)}", log_type="system")
        
        if not history_found:
            self.add_to_log(f"No training history found for {model_filename}", log_type="system")
            
            # Generate some dummy data for visualization
            if model_info["episodes"] > 0:
                # Generate dummy scores with an upward trend
                scores = np.linspace(0, 150, model_info["episodes"])
                scores = scores + np.random.normal(0, 20, model_info["episodes"])
                scores = np.clip(scores, 0, None)
                
                # Generate running averages
                window_size = min(100, len(scores))
                running_avgs = pd.Series(scores).rolling(window=window_size).mean().tolist()
                
                # Generate dummy losses
                losses = np.linspace(10, 0.1, 1000) + np.random.normal(0, 0.5, 1000)
                losses = np.clip(losses, 0.01, None)
                
                # Generate dummy Q-values
                q_values = np.linspace(0, 150, 1000) + np.random.normal(0, 15, 1000)
        
        # Update the training graph
        self.update_training_graph(scores, running_avgs, losses, q_values)

    def delete_model(self):
        """Delete the selected model after confirmation."""
        selected_items = self.models_tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Please select a model first.")
            return
        
        # Get the selected model
        item_id = selected_items[0]
        item_values = self.models_tree.item(item_id, "values")
        model_filename = item_values[0]
        
        # Find the model info
        model_info = next((m for m in self.models_info if m["filename"] == model_filename), None)
        if not model_info:
            messagebox.showerror("Error", "Model information not found.")
            return
        
        # Ask for confirmation
        confirm = messagebox.askyesno(
            "Confirm Delete", 
            f"Are you sure you want to delete {model_filename}?\n\nThis action cannot be undone."
        )
        
        if confirm:
            try:
                # Delete the model file
                os.remove(model_info["path"])
                
                # Also delete the history file if it exists
                history_path = os.path.join(os.path.dirname(model_info["path"]), 
                                           f"{os.path.splitext(model_filename)[0]}_history.json")
                if os.path.exists(history_path):
                    os.remove(history_path)
                
                # Remove from treeview
                self.models_tree.delete(item_id)
                
                # Remove from models_info
                self.models_info = [m for m in self.models_info if m["filename"] != model_filename]
                
                # Log the deletion
                self.add_to_log(f"Deleted model: {model_filename}", log_type="system")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete model: {str(e)}")

    def on_model_select(self, event):
        """Handle model selection event."""
        selected_items = self.models_tree.selection()
        if not selected_items:
            return
        
        # Get the selected model
        item_id = selected_items[0]
        item_values = self.models_tree.item(item_id, "values")
        model_filename = item_values[0]
        
        # Update selected model variable
        self.selected_model.set(model_filename)
        
        # Automatically load the model stats
        self.load_model_stats()
    
    def on_model_type_changed(self, event=None):
        """Handle model type selection change."""
        model_type = self.model_type_var.get()
        
        if "Q-Learning" in model_type:
            self.model_info_label.config(
                text="Q-Learning: Tabular method, absolute actions, perfect memory",
                foreground="purple"
            )
        elif "Enhanced" in model_type:
            self.model_info_label.config(
                text="Enhanced: A* guidance, curriculum learning, trap detection",
                foreground="green"
            )
        else:
            self.model_info_label.config(
                text="Enhanced: A* guidance, curriculum learning, trap detection",
                foreground="green"
            )
    
    def on_stuck_detection_toggled(self):
        """Enable/disable stuck detection parameter controls."""
        is_enabled = self.stuck_detection_var.get()
        
        # Enable or disable all child widgets in stuck_params_frame
        for child in self.stuck_params_frame.winfo_children():
            for widget in child.winfo_children():
                widget_type = widget.winfo_class()
                if widget_type in ('TLabel', 'TScale', 'TEntry', 'TSpinbox'):
                    widget.config(state='normal' if is_enabled else 'disabled')
    
    def find_next_model_number(self):
        """Find the next available model number."""
        import glob
        import re
        
        # Find all enhanced DQN models with numbers
        pattern = os.path.join(self.model_dir, "snake_enhanced_dqn_*.pth")
        existing_models = glob.glob(pattern)
        
        if not existing_models:
            return 1
        
        # Extract numbers from filenames
        numbers = []
        for model_path in existing_models:
            filename = os.path.basename(model_path)
            match = re.search(r'snake_enhanced_dqn_(\d+)\.pth', filename)
            if match:
                numbers.append(int(match.group(1)))
        
        if not numbers:
            return 1
        
        return max(numbers) + 1
    
    def update_model_number_hint(self):
        """Update the model number hint label based on current selection."""
        model_num = self.model_number_var.get().strip()
        
        if model_num:
            # User has entered or selected a model number
            model_filename = f"snake_enhanced_dqn_{model_num}.pth"
            model_path = os.path.join(self.model_dir, model_filename)
            
            if os.path.exists(model_path):
                if self.use_checkpoint_var.get():
                    # New model checkbox is checked - will overwrite
                    hint_text = f"⚠ Will create NEW model #{model_num} (overwrites existing)"
                    hint_color = 'orange'
                else:
                    # Will continue training
                    hint_text = f"✓ Will continue training model #{model_num}"
                    hint_color = 'green'
            else:
                # Model doesn't exist
                hint_text = f"✓ Will create NEW model #{model_num}"
                hint_color = 'blue'
        else:
            # No model number specified - will use default
            default_model_path = os.path.join(self.model_dir, "snake_enhanced_dqn.pth")
            
            if os.path.exists(default_model_path):
                if self.use_checkpoint_var.get():
                    hint_text = f"⚠ Will create NEW default model (overwrites existing)"
                    hint_color = 'orange'
                else:
                    hint_text = f"✓ Will continue training default model"
                    hint_color = 'green'
            else:
                hint_text = f"✓ Will create NEW default model (next available: {self.next_model_number})"
                hint_color = 'blue'
        
        self.model_number_hint_label.config(text=hint_text, foreground=hint_color)

    def start_training(self):
        """Start the training process."""
        if self.is_training:
            messagebox.showinfo("Info", "Training is already in progress.")
            return
        
        try:
            # Get parameters from UI
            episodes = int(self.episodes_var.get())
            save_interval = int(self.save_interval_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            use_checkpoint = self.use_checkpoint_var.get()
            
            # Validate parameters
            if episodes <= 0 or save_interval <= 0 or batch_size <= 0 or learning_rate <= 0:
                messagebox.showerror("Error", "All parameters must be positive numbers.")
                return
            
            # Get model number if specified
            model_number_str = self.model_number_var.get().strip()
            model_number = None
            if model_number_str:
                try:
                    model_number = int(model_number_str)
                    if model_number <= 0:
                        messagebox.showerror("Error", "Model number must be a positive integer.")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Model number must be a valid integer.")
                    return
            
            # Determine which training script to use based on model type
            model_type = self.model_type_var.get()
            use_qlearning = "Q-Learning" in model_type
            use_enhanced = "Enhanced" in model_type
            
            if use_qlearning:
                # Use Q-Learning training script (train_qlearning.py)
                training_script_path = os.path.join(os.path.dirname(__file__), "train_qlearning.py")
                cmd = [
                    sys.executable,
                    training_script_path,
                    "--episodes", str(episodes),
                    "--save-interval", str(save_interval),
                    "--learning-rate", str(learning_rate),
                    "--batch-size", str(batch_size)
                ]
                
                self.add_to_log("=" * 50)
                self.add_to_log("Q-LEARNING TRAINING")
                self.add_to_log("=" * 50)
                self.add_to_log(f"Algorithm: Tabular Q-Learning")
                self.add_to_log(f"State Space: 11 features (discrete)")
                self.add_to_log(f"Action Space: 4 absolute actions (UP/DOWN/LEFT/RIGHT)")
                self.add_to_log(f"Learning Rate (alpha): {learning_rate}")
                self.add_to_log(f"Batch Size: {batch_size} (experience replay)")
                
                # Check if model exists
                qlearning_model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
                if os.path.exists(qlearning_model_path):
                    self.add_to_log(f"[OK] Continuing from existing Q-table")
                else:
                    self.add_to_log(f"[NEW] Starting with fresh Q-table")
                
            elif use_enhanced:
                # Use the enhanced training script (train_enhanced.py)
                training_script_path = os.path.join(os.path.dirname(__file__), "train_enhanced.py")
                cmd = [
                    sys.executable,
                    training_script_path,
                    "--episodes", str(episodes),
                    "--save-interval", str(save_interval),
                    "--batch-size", str(batch_size),
                    "--learning-rate", str(learning_rate)
                ]
                
                # Add model number if specified
                if model_number is not None:
                    cmd.extend(["--model-number", str(model_number)])
                    self.add_to_log(f"Using model number: {model_number}")
                
                # Add stuck detection parameters
                if hasattr(self, 'stuck_detection_var'):
                    if self.stuck_detection_var.get():
                        cmd.extend([
                            "--enable-stuck-detection",
                            "--stuck-sensitivity", str(self.stuck_sensitivity_var.get()),
                            "--stuck-cooldown", str(self.stuck_cooldown_var.get()),
                            "--stuck-boost", str(self.stuck_boost_var.get()),
                            "--stuck-improvement", str(self.stuck_improvement_var.get())
                        ])
                        self.add_to_log(f"Stuck Detection: ENABLED (sensitivity={self.stuck_sensitivity_var.get()}, cooldown={self.stuck_cooldown_var.get()} eps)")
                    else:
                        cmd.append("--disable-stuck-detection")
                        self.add_to_log(f"Stuck Detection: DISABLED")
                
                # Enhanced model: checkbox checked = new model, unchecked = continue training
                if use_checkpoint:
                    cmd.append("--new-model")
                    if model_number:
                        self.add_to_log(f"Starting fresh Enhanced DQN model #{model_number}")
                    else:
                        self.add_to_log("Starting with fresh Enhanced DQN model")
                else:
                    if model_number:
                        self.add_to_log(f"Continuing training from Enhanced DQN model #{model_number}")
                    else:
                        self.add_to_log("Continuing training from existing Enhanced DQN checkpoint")
                
                self.add_to_log(f"Using Enhanced DQN (34 features, A* reward shaping, curriculum learning)")
                self.add_to_log(f"NOTE: Model architecture updated to 34 features - incompatible with old 31-feature models")
                
            # Start the training process
            self.add_to_log(f"Starting training with parameters: {' '.join(cmd[2:])}")
            
            # Set environment variable to force unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Use subprocess with Popen to get output
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env  # Pass the environment with unbuffered flag
            )
            
            # Start a thread to read the output
            self.is_training = True
            threading.Thread(target=self.read_process_output, daemon=True).start()
            
            # Update UI
            self.status_label.config(text="Training in progress")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
            self.add_to_log(f"Error starting training: {str(e)}")

    def stop_training(self):
        """Stop the training process."""
        if not self.is_training or not self.training_process:
            return
            
        try:
            # Ask for confirmation
            confirm = messagebox.askyesno(
                "Confirm Stop", 
                "Are you sure you want to stop the training?\n\nThe model will be saved at its current state."
            )
            
            if confirm:
                self.add_to_log("Stopping training process...")
                
                # Before terminating, try to save the training history
                self.save_training_data()
                
                # Send terminate signal to process
                self.training_process.terminate()
                
                # Update UI
                self.status_label.config(text="Stopping...")
                
                # Give it a moment for cleanup
                self.root.after(2000, self.finalize_stop)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop training: {str(e)}")
            self.add_to_log(f"Error stopping training: {str(e)}")
            
    def save_training_data(self):
        """Save training data collected so far to ensure it's not lost when stopping."""
        try:
            if not hasattr(self, 'training_data') or not self.training_data['scores']:
                self.add_to_log("No training data to save.")
                return False
            
            # Create filename based on current model or timestamp
            model_name = self.selected_model.get()
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snake_dqn_model_ep{len(self.training_data['scores'])}_interrupted_{timestamp}"
            else:
                # Remove extension if present
                model_basename = os.path.splitext(model_name)[0]
                filename = f"{model_basename}_history"
            
            # Prepare history data
            history = {
                "scores": self.training_data['scores'],
                "running_avgs": self.training_data['running_avgs'],
                "steps": self.training_data['steps'],
                "best_score": self.training_data['best_score'],
                "episodes_completed": len(self.training_data['scores']),
                "training_interrupted": True,
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            # Save to file
            history_path = os.path.join(self.model_dir, f"{filename}.json")
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.add_to_log(f"Training history saved to {history_path}", log_type="system")
            return True
        except Exception as e:
            self.add_to_log(f"Error saving training history: {str(e)}", log_type="system")
            return False

    def finalize_stop(self):
        """Finalize the training stop process."""
        self.is_training = False
        self.training_process = None 
        self.status_label.config(text="Ready")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.add_to_log("Training stopped.")
        
        # Rescan models to show the latest
        self.scan_models()

    def read_process_output(self):
        """Read and process the output from the training process."""
        try:
            # Initialize training data storage
            if not hasattr(self, 'training_data'):
                self.training_data = {
                    'scores': [],
                    'running_avgs': [],
                    'steps': [],
                    'best_score': 0,
                    'losses': [],
                    'q_values': [],
                    'epsilon_values': [],  # Track actual epsilon values
                    'lr_values': []  # Track actual learning rate values
                }
                
            # Initialize training start time
            self.training_start_time = time.time()
            
            # Track whether training data has been saved
            training_data_saved = False
            
            # Add a log entry to confirm we're reading output
            self.root.after(0, lambda: self.add_to_log("Reading training output..."))
                
            for line in iter(self.training_process.stdout.readline, ''):
                if not line:
                    break
                    
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
                # Parse line for training stats FIRST (this will handle episode updates)
                self.parse_training_output(line_stripped)
                
                # Add all non-empty lines to the log (episode lines will be added as status updates by parse_training_output)
                if not line_stripped.startswith("DQN Episode:") and not line_stripped.startswith("Enhanced DQN Episode:"):
                    # Classify the message type to route to appropriate log
                    log_type = "training"
                    if any(keyword in line_stripped for keyword in ["saved", "Checkpoint", "GPU:", "CUDA", "device:", "Model loaded"]):
                        log_type = "system"
                    
                    # Use after to ensure thread-safe GUI update
                    self.root.after(0, lambda msg=line_stripped, lt=log_type: self.add_to_log(msg, log_type=lt))
                
                # Look for checkpoint saves to update our saved history
                if "Checkpoint saved" in line_stripped or "Model saved" in line_stripped:
                    # Save the training data when a checkpoint is saved
                    self.root.after(0, self.save_training_data)
                    training_data_saved = True
            
            # Process has ended
            return_code = self.training_process.wait()
            
            # Make sure training data is saved if training completes successfully
            if return_code == 0:
                self.root.after(0, lambda: self.add_to_log("Training completed successfully."))
                if not training_data_saved:
                    self.root.after(0, self.save_training_data)
            else:
                self.root.after(0, lambda rc=return_code: self.add_to_log(f"Training process ended with return code {rc}"))
                # Try to save partial data even on error
                if not training_data_saved:
                    self.root.after(0, self.save_training_data)
                
            # Update UI
            self.root.after(0, self.finalize_stop)
            
        except Exception as e:
            error_msg = f"Error reading training output: {str(e)}"
            self.root.after(0, lambda: self.add_to_log(error_msg))
            # Try to save data even on exception
            self.root.after(0, self.save_training_data)
            self.root.after(0, self.finalize_stop)

    def parse_training_output(self, line):
        """Parse training output to update graphs."""
        # Check for curriculum advancement messages
        # Format: [CURRICULUM] ADVANCED: Stage 1 -> Stage 2
        curriculum_match = re.search(r'\[CURRICULUM\] ADVANCED: Stage (\d+) -> Stage (\d+)', line)
        if curriculum_match:
            old_stage = int(curriculum_match.group(1))
            new_stage = int(curriculum_match.group(2))
            
            # Record the advancement with the current episode number
            if hasattr(self, 'training_data') and self.training_data['scores']:
                current_episode = len(self.training_data['scores'])
                self.curriculum_advancements.append((current_episode, old_stage, new_stage))
                
                # Add special log message
                advancement_msg = f"[CURRICULUM] ADVANCEMENT: Stage {old_stage} -> {new_stage} at Episode {current_episode}"
                self.root.after(0, lambda msg=advancement_msg: self.add_to_log(msg, log_type="training"))
        
        # Parse episode info - handles both integer and floating point scores
        # Format: Enhanced DQN Episode: 100/1000, Score: 25.5, Steps: 45, Best: 35.2, Avg: 15.5, Epsilon: 0.9, LR: 0.00350, Curriculum: Stage 1, A*: 0.50, Time: 120.5s
        match = re.search(r'(?:Enhanced )?DQN Episode: (\d+)/(\d+), Score: ([\d.eE+-]+), Steps: (\d+), Best: ([\d.eE+-]+), Avg: ([\d.]+), Epsilon: ([\d.]+)(?:, LR: ([\d.]+))?(?:, Curriculum: Stage (\d+))?(?:, A\*: ([\d.]+))?, Time: ([\d.]+)s', line)
        
        # Also check for Q-Learning output format
        # Format: "Episode 100 | Score: 25 | Steps: 45 | Avg: 15.50 | epsilon: 0.9500 | Q-states: 500 | Best: 35"
        qlearning_match = None
        if not match:
            qlearning_match = re.search(r'Episode\s+(\d+)\s*\|\s*Score:\s*(\d+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*Avg:\s*([\d.]+)\s*\|\s*epsilon:\s*([\d.]+)\s*\|\s*Q-states:\s*(\d+)\s*\|\s*Best:\s*(\d+)', line)
        
        if match:
            episode = int(match.group(1))
            total_episodes = int(match.group(2))
            score = float(match.group(3))
            steps = int(match.group(4))
            best = float(match.group(5))
            avg = float(match.group(6))
            epsilon = float(match.group(7))
            lr = float(match.group(8)) if match.group(8) else None  # NEW: Extract LR
            curriculum_stage = int(match.group(9)) if match.group(9) else None
            astar_prob = float(match.group(10)) if match.group(10) else None
            time_taken = float(match.group(11))
            
            # Update progress in UI (using lambda to capture current values)
            self.root.after(0, lambda e=episode, t=total_episodes: self.progress_label.config(text=f"{e}/{t}"))
            self.root.after(0, lambda b=best: self.score_label.config(text=f"{b:.1f}"))
            
            # Create a more informative training summary entry
            summary = f"Episode: {episode}/{total_episodes} | Score: {score:.1f} | Best: {best:.1f} | Avg: {avg:.2f} | Steps: {steps} | Epsilon: {epsilon:.4f}"
            if lr is not None:
                summary += f" | LR: {lr:.5f}"  # NEW: Add LR to summary
            if curriculum_stage is not None:
                summary += f" | Curriculum: Stage {curriculum_stage}"
            if astar_prob is not None:
                summary += f" | A*: {astar_prob:.2f}"
            summary += f" | Time: {time_taken:.2f}s"
            # Use root.after to ensure thread-safe GUI update
            # Don't use is_status=True to allow each episode to be logged on a new line
            self.root.after(0, lambda s=summary: self.add_to_log(f"Training Status: {s}", is_status=False))
            
            # Track training data to ensure we have it for saving later
            if not hasattr(self, 'training_data'):
                self.training_data = {
                    'scores': [],
                    'running_avgs': [],
                    'steps': [],
                    'best_score': 0,
                    'epsilon_values': [],  # Track actual epsilon values
                    'lr_values': []  # NEW: Track actual learning rate values
                }
            
            self.training_data['scores'].append(score)
            self.training_data['running_avgs'].append(avg)
            self.training_data['steps'].append(steps)
            self.training_data['best_score'] = max(self.training_data['best_score'], best)
            self.training_data['epsilon_values'].append(epsilon)  # Store actual epsilon
            if lr is not None:
                # Ensure lr_values key exists (for compatibility with older code)
                if 'lr_values' not in self.training_data:
                    self.training_data['lr_values'] = []
                self.training_data['lr_values'].append(lr)  # Store actual LR
        
        elif qlearning_match:
            # Handle Q-Learning output format
            episode = int(qlearning_match.group(1))
            score = int(qlearning_match.group(2))
            steps = int(qlearning_match.group(3))
            avg = float(qlearning_match.group(4))
            epsilon = float(qlearning_match.group(5))
            q_states = int(qlearning_match.group(6))
            best = int(qlearning_match.group(7))
            
            # Update progress in UI
            self.root.after(0, lambda e=episode: self.progress_label.config(text=f"{e}"))
            self.root.after(0, lambda b=best: self.score_label.config(text=f"{b}"))
            
            # Create training summary
            summary = f"Episode: {episode} | Score: {score} | Best: {best} | Avg: {avg:.2f} | Steps: {steps} | Epsilon: {epsilon:.4f} | Q-states: {q_states}"
            self.root.after(0, lambda s=summary: self.add_to_log(f"Q-Learning Status: {s}", is_status=False))
            
            # Track training data
            if not hasattr(self, 'training_data'):
                self.training_data = {
                    'scores': [],
                    'running_avgs': [],
                    'steps': [],
                    'best_score': 0,
                    'epsilon_values': [],
                    'q_states': []
                }
            
            self.training_data['scores'].append(score)
            self.training_data['running_avgs'].append(avg)
            self.training_data['steps'].append(steps)
            self.training_data['best_score'] = max(self.training_data['best_score'], best)
            self.training_data['epsilon_values'].append(epsilon)
            
            # Track Q-table size for Q-Learning
            if 'q_states' not in self.training_data:
                self.training_data['q_states'] = []
            self.training_data['q_states'].append(q_states)
            
            # Update graphs (Q-Learning doesn't have losses/q_values like DQN)
            self.root.after(0, lambda s=[score], a=[avg]: self.update_training_graph(scores=s, running_avgs=a))
    
    def add_to_log(self, message, is_status=False, log_type="training"):
        """Add a message to the training or system log.
        
        Args:
            message: The message to add
            is_status: If True, will replace the last status message rather than adding a new line
            log_type: "training" for training log, "system" for system log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Select the appropriate log widget
        if log_type == "system":
            log_widget = self.system_log_text
            # Add to system log list
            if not hasattr(self, 'system_log'):
                self.system_log = []
            self.system_log.append(log_entry)
            if len(self.system_log) > 1000:
                self.system_log = self.system_log[-1000:]
        else:
            log_widget = self.log_text
            # Add to training log list (always append to history)
            self.training_log.append(log_entry)
            # Keep log size manageable
            if len(self.training_log) > 1000:
                self.training_log = self.training_log[-1000:]
        
        # Update log text widget
        log_widget.config(state=tk.NORMAL)
        
        # For status updates, replace the last status line if it exists (only for training log)
        if is_status and log_type == "training" and hasattr(self, 'last_status_position'):
            # Delete the last status line
            log_widget.delete(self.last_status_position, self.last_status_position + "+1line")
            # Insert the new status at the same position
            log_widget.insert(self.last_status_position, log_entry)
            # Remember the position again
            self.last_status_position = self.last_status_position
        else:
            # For normal messages, just append
            log_widget.insert(tk.END, log_entry)
            # If this is a new status message, remember its position
            if is_status and log_type == "training":
                self.last_status_position = log_widget.index(tk.END + "-1line")
        
        # Auto-scroll to bottom and disable editing
        log_widget.see(tk.END)
        log_widget.config(state=tk.DISABLED)

    def update_memory_usage(self):
        """Update the memory usage display."""
        if torch.cuda.is_available():
            try:
                # Get GPU memory usage
                allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                memory_text = f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f}/{total:.2f} GB reserved"
                self.memory_label.config(text=memory_text)
            except:
                self.memory_label.config(text="VRAM: Error reading GPU memory")
        else:
            try:
                # Get system RAM usage
                ram = psutil.virtual_memory()
                memory_text = f"RAM: {ram.used/(1024**3):.2f}/{ram.total/(1024**3):.2f} GB ({ram.percent}%)"
                self.memory_label.config(text=memory_text)
            except:
                self.memory_label.config(text="RAM: Error reading system memory")
                
    def check_cuda_setup(self):
        """Run CUDA setup check and display results."""
        import subprocess
        import sys
        
        # Show a message that we're checking
        messagebox.showinfo(
            "CUDA Check", 
            "Running CUDA compatibility check...\n"
            "This will open a terminal window with detailed information."
        )
        
        # Run the check_cuda.py script
        check_cuda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_cuda.py")
        
        try:
            # Run the check_cuda.py script in a separate process
            subprocess.Popen([sys.executable, check_cuda_path], 
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run CUDA check: {str(e)}")
    
    def get_dynamic_update_interval(self):
        """
        Calculate dynamic update interval based on training progress.
        Reduces update frequency as training progresses to improve performance.
        """
        if not hasattr(self, 'training_data') or not self.training_data.get('scores'):
            return 2000  # Default 2 seconds for early training
        
        episode_count = len(self.training_data['scores'])
        
        # Dynamic intervals based on episode count
        if episode_count < 50:
            return 1000  # 1 second - fast feedback for early training
        elif episode_count < 150:
            return 2000  # 2 seconds
        elif episode_count < 300:
            return 3000  # 3 seconds
        elif episode_count < 500:
            return 5000  # 5 seconds
        else:
            return 8000  # 8 seconds - very long training runs

    def update_ui(self):
        """Periodically update the UI elements with performance optimizations."""
        # OPTIMIZATION 1: Skip updates if window is minimized/hidden
        try:
            window_state = self.root.state()
            if window_state in ('iconic', 'withdrawn'):
                # Window minimized/hidden, skip expensive updates
                update_interval = self.get_dynamic_update_interval()
                self.root.after(update_interval, self.update_ui)
                return
        except:
            pass  # Continue if state check fails
        
        # Update memory usage (lightweight)
        self.update_memory_usage()
        
        # OPTIMIZATION 2: Only update graph if training is active AND new data arrived
        if self.is_training and hasattr(self, 'training_data') and self.training_data['scores']:
            current_episode_count = len(self.training_data['scores'])
            
            # Only update if new data arrived
            if current_episode_count > self.last_episode_count:
                self.last_episode_count = current_episode_count
                
                # OPTIMIZATION 3: Decide between incremental update vs full redraw
                # Full redraw every 50 episodes OR if curriculum just advanced
                should_full_redraw = (
                    current_episode_count - self.last_full_redraw >= 50 or
                    current_episode_count < 10  # Always full redraw for first 10 episodes
                )
                
                # Update the graph with current training data
                self.update_training_graph(
                    scores=self.training_data['scores'],
                    running_avgs=self.training_data['running_avgs'],
                    losses=self.training_data.get('losses', []),
                    q_values=self.training_data.get('q_values', []),
                    force_full_redraw=should_full_redraw
                )
                
                if should_full_redraw:
                    self.last_full_redraw = current_episode_count
        
        # OPTIMIZATION 4: Schedule next update with dynamic interval
        update_interval = self.get_dynamic_update_interval()
        self.root.after(update_interval, self.update_ui)

def main():
    """Main entry point for the Training UI."""
    root = tk.Tk()
    app = TrainingUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()