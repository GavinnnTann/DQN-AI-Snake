# Training UI Model Selection Fix

**Date:** October 14, 2025  
**Issue:** Unable to select specific numbered models (e.g., model 2 or 3) for continued training  
**Impact:** Training always defaulted to the base model (`snake_enhanced_dqn.pth`) regardless of selection

---

## 🐛 **Problem Description**

When users selected a numbered model (like `snake_enhanced_dqn_3.pth`) from the model browser to continue training, the UI would:

1. ✅ Display the model's statistics correctly
2. ✅ Show the model in the browser
3. ❌ **BUT** not pass the model number to the training script
4. ❌ **Result:** Training saved to default `snake_enhanced_dqn.pth` instead of `snake_enhanced_dqn_3.pth`

### Root Cause

The UI's `load_model_stats()` function did not extract and populate the model number from the selected filename, so `train_enhanced.py` received `--model-number None` and defaulted to the base model.

---

## ✅ **Solution Implemented**

### 1. **Auto-populate Model Number on Selection**

Modified `load_model_stats()` in `training_ui.py` to:
- Extract model number from filename using regex (`_(\d+)\.pth$`)
- Auto-populate the "Model Number" field
- Log the selection to System Log

**Code Location:** `training_ui.py`, line ~2310

```python
# BUGFIX: Extract model number from filename and populate UI field
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
```

### 2. **Dynamic Hint Label**

Added a smart hint label that updates in real-time to show what will happen when training starts:

**Examples:**
- ✓ `Will continue training model #3` (green) - Model exists, checkbox unchecked
- ✓ `Will create NEW model #5` (blue) - Model doesn't exist
- ⚠ `Will create NEW model #3 (overwrites existing)` (orange) - Checkbox checked, model exists

**Code Location:** `training_ui.py`, lines ~1317-1324, ~2555-2596

### 3. **Real-time Feedback**

Added variable trace and checkbox callback to update the hint whenever:
- User manually types a model number
- User toggles the "Start from New Model" checkbox
- User selects a model from the browser

---

## 🧪 **Testing Instructions**

### Test Case 1: Select Existing Numbered Model
1. Open Training UI: `python train_snake.py`
2. In the model browser, double-click `snake_enhanced_dqn_3.pth`
3. ✅ **Expected:** Model Number field shows `3`
4. ✅ **Expected:** Hint shows "✓ Will continue training model #3" (green)
5. Click "Start Training"
6. ✅ **Expected:** Training saves to `snake_enhanced_dqn_3.pth`

### Test Case 2: Create New Numbered Model
1. Open Training UI
2. Type `5` in Model Number field
3. ✅ **Expected:** Hint shows "✓ Will create NEW model #5" (blue)
4. Click "Start Training"
5. ✅ **Expected:** Creates `snake_enhanced_dqn_5.pth`

### Test Case 3: Overwrite Existing Model
1. Open Training UI
2. Select `snake_enhanced_dqn_2.pth` from browser
3. Check "Start from New Model" checkbox
4. ✅ **Expected:** Hint shows "⚠ Will create NEW model #2 (overwrites existing)" (orange)
5. Click "Start Training"
6. ✅ **Expected:** Overwrites `snake_enhanced_dqn_2.pth` with fresh model

### Test Case 4: Continue Default Model
1. Open Training UI
2. Leave Model Number field empty
3. Ensure "Start from New Model" is unchecked
4. ✅ **Expected:** Hint shows "✓ Will continue training default model" (green)
5. Click "Start Training"
6. ✅ **Expected:** Continues training `snake_enhanced_dqn.pth`

---

## 📝 **Files Modified**

### `advanced_snake/training_ui.py`

**Line ~1317:** Added dynamic hint label
```python
self.model_number_hint_label = ttk.Label(...)
self.model_number_var.trace('w', lambda *args: self.update_model_number_hint())
```

**Line ~1300:** Added callback to checkbox
```python
command=self.update_model_number_hint
```

**Line ~2310:** Auto-populate model number on selection
```python
model_number_match = re.search(r'_(\d+)\.pth$', model_filename)
if model_number_match:
    self.model_number_var.set(model_number_match.group(1))
```

**Line ~2555:** New method `update_model_number_hint()`
```python
def update_model_number_hint(self):
    """Update the model number hint label based on current selection."""
    # ... (60 lines of logic)
```

---

## 🎯 **Benefits**

1. **✅ Correct Model Continuation** - Training now saves to the selected model
2. **✅ Clear User Feedback** - Users know exactly what will happen before starting
3. **✅ Prevents Accidents** - Warning when about to overwrite an existing model
4. **✅ Better UX** - Auto-population saves manual typing
5. **✅ Visual Indicators** - Color-coded hints (green/blue/orange) for quick scanning

---

## 🔄 **Workflow Examples**

### Continue Training Model #3:
```
1. Double-click "snake_enhanced_dqn_3.pth" in model browser
2. Verify hint: "✓ Will continue training model #3"
3. Click "Start Training"
4. Training continues from Episode 1001 → 2000 (or wherever it left off)
5. Saves to: snake_enhanced_dqn_3.pth
```

### Create New Model #7:
```
1. Type "7" in Model Number field
2. Check "Start from New Model" checkbox
3. Verify hint: "✓ Will create NEW model #7"
4. Click "Start Training"
5. Training starts from Episode 1
6. Saves to: snake_enhanced_dqn_7.pth
```

---

## 🚀 **Future Enhancements** (Optional)

1. **Model Comparison** - Side-by-side stats for multiple models
2. **Model Tagging** - Add custom tags/notes to models (e.g., "best-long-runs", "experimental")
3. **Auto-backup** - Create backup before overwriting
4. **Model Lineage** - Track which models were forked from which

---

**Status:** ✅ **FIXED** - Model selection now works correctly for continued training
