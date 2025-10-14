## Repo orientation — what matters for code changes

This repository is a self-contained Python project implementing a playable Snake game plus several AI agents and training tooling under `advanced_snake/`.
Key entry points:
- `play_snake.py` — launcher that adds `advanced_snake/` to PYTHONPATH and runs `advanced_snake/main.py` (game play).
- `train_snake.py` / `advanced_snake/training_ui.py` — GUI training interface.
- `advanced_snake/train_enhanced.py` — CLI training for the Enhanced DQN agent.

Core components and responsibilities
- `advanced_snake/game_engine.py`: game state, movement, collision detection, rendering hooks used by agents.
- `advanced_snake/algorithms.py`: A* and Dijkstra implementations used as guidance / baselines.
- `advanced_snake/enhanced_dqn.py`: Enhanced DQN agent, curriculum logic, reward shaping and model save/load.
- `advanced_snake/q_learning.py`: Tabular Q-learning agent (incompatible model format with DQN).
- `advanced_snake/main.py`: glue code — menu, model browser, input mapping, loads agents and ensures compatibility with saved models.
- `advanced_snake/constants.py`: single source of truth for hyperparameters, curriculum thresholds, stuck-detection and model paths.

Important conventions and gotchas (explicit and actionable)
- Action encoding (must match training and runtime): DQN uses relative actions mapping where
  `0 = turn right`, `1 = straight`, `2 = turn left`. This mapping appears in
  `advanced_snake/main.py::convert_relative_to_absolute_direction` and
  `advanced_snake/enhanced_dqn.py::perform_action`; never change one without the other.
- State size differences:
  - Classic DQN / `game_engine.get_state()` returns an 11-feature tensor (used by Q-learning and original DQN).
  - Enhanced DQN expects a 34-feature tensor produced by `EnhancedStateRepresentation.get_enhanced_state()` (in `enhanced_dqn.py`).
  Agents and saved models are not cross-compatible (Enhanced DQN ≠ Original DQN ≠ Q-Learning).
- Models and history are stored under `advanced_snake/models/` with naming patterns like
  `snake_enhanced_dqn.pth` and `snake_enhanced_dqn_history.json`, or numbered variants `snake_enhanced_dqn_1.pth`.
  `train_enhanced.py` will auto-detect and resume training using the history file.

Developer workflows (commands and examples)
- Install deps: `pip install -r requirements.txt` (project root).
- Play locally: `python play_snake.py` — launches `advanced_snake/main.py` via the launcher.
- Training (UI recommended): `python train_snake.py` or `python advanced_snake/training_ui.py`.
- Headless CLI training (enhanced):
  `cd advanced_snake && python train_enhanced.py --episodes 1000 --new-model` (use `--new-model` to start fresh).
- Resume training: `train_enhanced.py` auto-loads `*_history.json` and continues from saved `episodes_completed`.

Integration points and diagnostics for debugging
- GPU detection: `training_ui.py` and various scripts use `torch.cuda.is_available()`; if CUDA appears missing the UI tries `nvidia-smi`.
  To confirm GPU status: `python check_cuda.py` (comprehensive diagnostics with --detailed, --monitor, --fix flags).
  Quick manual check: `python -c "import torch; print(torch.cuda.is_available())"`.
- Model load errors will print stack traces; check `advanced_snake/train_enhanced.py` logs for paths it attempted and the history JSON (`models/*.json`).
- PyTorch 2.6+ compatibility: models saved with older PyTorch may fail to load due to `weights_only=True` default. The codebase dynamically adds all numpy dtype classes to safe globals in `main.py` and `enhanced_dqn.py` load methods to handle this.
- When modifying reward, curriculum or epsilon decay, update both `constants.py` (global knobs) and `enhanced_dqn.py` (uses/overrides those constants) to keep behavior consistent.

Files you will most often change and why
- `constants.py` — hyperparameter tuning, curriculum thresholds, stuck-detection parameters.
- `enhanced_dqn.py` — agent architecture, reward shaping, curriculum advancement rules.
- `training_ui.py` — wiring and visualization; change here only for UI/UX improvements.
- `game_engine.py` / `algorithms.py` — modify when changing core rules (movement, valid-move semantics) because many agents rely on them.

Small examples (copyable) — do not break these contracts
- Relative action mapping (training ↔ runtime):
  - `advanced_snake/main.py` expects the agent's action index to follow: `0=right, 1=straight, 2=left`.
  - `advanced_snake/enhanced_dqn.py::perform_action` implements the same mapping — keep them aligned.
- Model resume: `train_enhanced.py` reads `{model}_history.json` and uses `episodes_completed` to compute `start_episode`.

Scope for agents: focus edits on the `advanced_snake/` folder. When touching saved-model formats, update loader/saver in `enhanced_dqn.py` and `main.py` together to avoid runtime incompatibility.

If anything above is unclear or you'd like the instructions expanded (examples for modifying curriculum, or a short checklist for adding a new agent), tell me which area to expand and I will iterate.
