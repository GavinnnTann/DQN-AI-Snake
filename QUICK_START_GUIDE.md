# Advanced Snake Game - Quick Start Guide

**New to the project? Start here!**

---

## 30-Second Start

```bash
# Clone and navigate to project
cd "Snake Game/advanced_snake"

# Install dependencies
pip install torch numpy matplotlib pygame

# Launch Training UI
python training_ui.py

# Click "Start Training" button
# Watch the magic happen! ✨
```

---

## 5-Minute Tutorial

### 1. Understanding the Game Modes

**Manual Mode** - You play with WASD keys  
**A* / Dijkstra** - Watch pathfinding algorithms  
**Q-Learning** - Tabular reinforcement learning  
**Enhanced DQN** - Deep learning with neural networks (BEST!)

### 2. Train Your First Agent

**Option A: Training UI (Recommended)**
```bash
python training_ui.py
```
1. Click "Start Training"
2. Watch Score Progression graph
3. Look for green gradient boxes (learning momentum)
4. Wait for ~500 episodes or until "Recent" gradient turns yellow
5. Model auto-saves to `models/` folder

**Option B: Command Line**
```bash
python train_enhanced.py --episodes 500
```

### 3. Watch Your Agent Play

```bash
python main.py
```
1. Select "Advanced DQN"
2. Choose "Load Existing Model"
3. Pick latest model (highest episode number)
4. Watch it play!

### 4. Understanding the Results

**In Training UI, look at the Score Progression graph:**

**Gradient Indicators (colored boxes in upper right):**
- 🟢 **Green** = Agent is learning (good!)
- 🟡 **Yellow** = Agent plateaued (might be done)
- 🔴 **Red** = Agent regressing (stop training!)

**Cyan stars** = Curriculum stage advancement (agent got better!)

**Gold star** = Best score achieved

---

## Understanding Key Concepts

### Curriculum Learning (5 Stages)

Think of it like school grades:

| Stage | Avg Score | Skill Level | What Agent Learns |
|-------|-----------|-------------|-------------------|
| 0 | 0-20 | Kindergarten | "Don't hit walls!" |
| 1 | 20-50 | Elementary | "Collect food consistently" |
| 2 | 50-100 | Middle School | "Avoid trapping myself" |
| 3 | 100-200 | High School | "Plan ahead" |
| 4 | 200+ | College | "Maximize score strategically" |

**Stage advancement** = Agent mastered current level, moves to next

### Gradient Indicators (Learning Momentum)

Three colored boxes show how fast agent is improving:

1. **Overall** - Total learning progress (start to now)
2. **Mid-term** - Recent acceleration (halfway point to now)
3. **Recent** - **MOST IMPORTANT** - Last 100 episodes

**What colors mean:**
- 🟢 Bright/Light Green = Improving (points per episode > 0.01)
- 🟡 Yellow = Stagnant (near zero change) → Training likely done
- 🔴 Red = Getting worse → STOP!

### Stuck Detection

**What it does:** Automatically boosts exploration (epsilon) when agent plateaus

**When to use:**
- ✅ Agent stuck at plateau for 200+ episodes
- ✅ Recent gradient Yellow, boost changes it to Green

**When to disable:**
- ❌ Recent gradient stays Yellow despite boosts (not helping)
- ❌ Training becomes unstable (scores jumping wildly)

**How to disable:**
In Training UI → Stuck Detection Controls → Uncheck "Enable Stuck Detection"

---

## Typical Training Timeline

**What to expect:**

```
Episodes 1-50:    Agent learns not to die immediately
                  Stage 0 → Beginner
                  Gradient: Bright Green (fast learning!)

Episodes 50-200:  Agent learns to collect food consistently
                  Stage 0→1 advancement around ep 100-150
                  Gradient: Green (good progress)

Episodes 200-500: Agent improves strategy
                  Stage 1→2 advancement around ep 300-400
                  Gradient: Light Green (slower but steady)

Episodes 500-1000: Agent refines tactics
                   Stage 2→3 advancement possible
                   Gradient: May turn Yellow (plateau)

Episodes 1000+:   Fine-tuning
                  Gradient: Usually Yellow (minimal improvement)
                  Consider stopping unless still Green
```

---

## Common Questions

**Q: How long should I train?**  
A: Until Recent gradient (bottom box) turns Yellow for 200+ episodes. Usually 500-1500 episodes.

**Q: My agent isn't improving. Why?**  
A: Check Recent gradient:
- Green → Still learning, keep going
- Yellow → Likely done, save and stop
- Red → Problem! Lower learning rate or reduce batch size

**Q: Should I use stuck detection?**  
A: Try it first (enabled by default). If Recent gradient doesn't improve after boosts, disable it.

**Q: What's a good final score?**  
A: 
- Stage 2 (50-100 avg): Decent
- Stage 3 (100-200 avg): Good
- Stage 4 (200+ avg): Excellent!

**Q: GPU or CPU?**  
A: GPU is 3-5x faster! But CPU works fine, just takes longer.

**Q: Can I resume training?**  
A: Yes! Load a checkpoint and keep training. Models saved every 100 episodes.

---

## Quick Troubleshooting

**Problem: Training very slow**
```bash
# Check if GPU is detected
python check_cude.py
# Extensive CUDA version Mapping and automated installation
# If False, install CUDA-enabled PyTorch
# If True but still slow, reduce batch size in constants.py
```

**Problem: All gradients Yellow but agent still bad (low scores)**
```python
# In constants.py, lower curriculum thresholds:
curriculum_thresholds = [15, 40, 80, 150]  # Instead of [20, 50, 100, 200]
```

**Problem: Training UI frozen/laggy**
```
Already fixed in latest version!
Performance optimized: 90% faster updates
Just update to latest training_ui.py
```

**Problem: Scores jumping wildly**
```python
# In enhanced_dqn.py, reduce learning rates:
stage_learning_rates = {
    0: 0.003,   # Was 0.005
    1: 0.002,   # Was 0.003
    2: 0.001,   # Was 0.002
    ...
}
```

---

## Essential Files

**To run:**
- `training_ui.py` → Launch training with visualization
- `train_enhanced.py` → Train without UI (faster)
- `main.py` → Play manually or watch agent

**To configure:**
- `constants.py` → All parameters (learning rate, epsilon, etc.)
- `enhanced_dqn.py` → Curriculum thresholds, architecture

**To learn:**
- `COMPLETE_REFERENCE_GUIDE.md` → Full documentation (THIS IS COMPREHENSIVE!)
- `GRADIENT_INDICATORS_GUIDE.md` → Gradient details
- `STUCK_DETECTION_TUNING_GUIDE.md` → Stuck detection tuning

---

## Next Steps

After training your first agent:

### 1. Experiment with Parameters
```python
# In constants.py, try:
STUCK_BOOST_COOLDOWN = 400  # Longer between boosts
STUCK_EPSILON_BOOST = 0.06  # Gentler boost

# Or disable stuck detection entirely:
ENABLE_STUCK_DETECTION = False
```

### 2. Tune Curriculum
```python
# In enhanced_dqn.py, adjust thresholds:
self.curriculum_thresholds = [30, 75, 150, 300]  # Harder stages
# or
self.curriculum_thresholds = [15, 40, 80, 150]   # Easier stages
```

### 3. Compare Training Runs
- Train with stuck detection enabled → Note max score
- Train with stuck detection disabled → Compare
- Train with different curriculum thresholds → Find optimal

### 4. Analyze Results
- Use gradient indicators to understand learning patterns
- Check which curriculum stages take longest
- Identify when training should stop (Yellow gradients)

---

## Command Cheat Sheet

```bash
# TRAINING
python training_ui.py                    # UI training (best for monitoring)
python train_enhanced.py --episodes 1000 # CLI training (faster)

# PLAYING
python main.py                           # Launch game menu

# TESTING
python test_gpu_usage.py                 # Check GPU setup
python -m py_compile *.py                # Syntax check all files

# TRAINING WITH OPTIONS
python train_enhanced.py \
  --episodes 2000 \
  --save-interval 200 \
  --disable-stuck-detection

# RESUME TRAINING
python train_enhanced.py \
  --episodes 1000 \
  --resume models/snake_dqn_model_ep500.pth
```

---

## Key Metrics to Watch

**During Training:**
1. **Recent Gradient** (bottom colored box)
   - Most important indicator
   - Green = keep training
   - Yellow = consider stopping
   
2. **Curriculum Stage** (shown in status/log)
   - Higher = better skill level
   - Stage 3-4 = excellent

3. **Average Score** (red line on graph)
   - Smooth upward trend = good
   - Flat for 200+ episodes = plateau

4. **Epsilon Boosts** (if enabled)
   - Should be infrequent (every 200+ episodes)
   - If too frequent → Consider disabling

**After Training:**
- **Max Score Achieved** (gold star on graph)
- **Final Average** (last 100 episodes)
- **Gradient Colors** (all Yellow = finished, Green = could continue)

---

## Decision Trees

### Should I Stop Training?

```
Is Recent gradient Yellow?
├─ No (Green/Red) → Continue training
└─ Yes
   └─ Has it been Yellow for 200+ episodes?
      ├─ No → Continue training
      └─ Yes
         └─ Is average score acceptable?
            ├─ Yes → STOP, save model
            └─ No → Try adjusting parameters, train more
```

### Should I Disable Stuck Detection?

```
Are epsilon boosts visible on graph?
├─ No → Keep enabled, might trigger later
└─ Yes
   └─ Does Recent gradient improve after boosts?
      ├─ Yes (Yellow→Green) → Keep enabled, it's working!
      └─ No (stays Yellow) → DISABLE, not helping
```

### Which Curriculum Threshold to Use?

```
Is agent stuck at Stage 0-1 for 300+ episodes?
├─ Yes → LOWER thresholds [15, 40, 80, 150]
└─ No
   └─ Is agent advancing too quickly?
      ├─ Yes → RAISE thresholds [30, 75, 150, 300]
      └─ No → Keep defaults [20, 50, 100, 200]
```

---

## Success Checklist

After training, you should see:

- ✅ At least Stage 2 achieved (avg score 50+)
- ✅ Gradients mostly Green early, Yellow late
- ✅ Curriculum advancements (cyan stars on graph)
- ✅ Max score significantly higher than average
- ✅ Model files saved in `models/` directory
- ✅ Agent can play reasonably well when loaded

If missing any:
- Review gradient indicators
- Check log for errors
- Ensure sufficient training episodes (500+)
- Verify GPU is being used (if available)

---
