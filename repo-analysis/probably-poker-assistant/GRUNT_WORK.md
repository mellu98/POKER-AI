# Poker AI Assistant - Grunt Work for Other Agents

**Status:** V1 Feature Complete. Most tasks below are DONE. This document lists remaining tasks for other AI agents.

**CRITICAL RULES:**
1. NO PLACEHOLDER DATA - Use real GTO ranges, real solver output, real values
2. Test every change against the existing test suite
3. Maintain the existing interfaces - don't change method signatures without updating callers

---

## Completed Tasks ✅

### Task 1: Generate Preflop GTO Ranges ✅ COMPLETE
**Files Created:**
- `database/preflop/open_ranges.json` - Full opening ranges for all positions
- `database/preflop/3bet_ranges.json` - 3bet/call/fold vs all positions
- `database/preflop/4bet_ranges.json` - 4bet ranges vs 3bets
- `tools/generate_gto_ranges.py` - Range generation script

**Note:** Used programmatic GTO-approximation algorithm rather than TexasSolver.

---

## High Priority Tasks (Remaining)

### Task 2: Add Community Card Regions to Anchor Config
**Estimated Complexity:** Low
**Files to Modify:**
- `config/anchor_config.json`

**Instructions:**
1. Add these regions to the `regions` object:
   - `community_cards`: Region containing all 5 community card positions
   - `pot_amount`: Region showing pot size
   - `player_stack`: Region showing player's chip stack
   - `current_bet`: Region showing current bet to call

**Note:** Exact pixel values depend on table resolution. User will need to calibrate.

---

### Task 3: Create Postflop Strategy Module ✅ COMPLETE
**Files Created:**
- `src/strategy/postflop_strategy.py` - Texture-aware postflop decisions
- `src/strategy/board_analyzer.py` - Board texture classification

**Implemented:**
- Board texture classification (dry/wet/medium/dynamic)
- C-bet strategy based on texture, hand strength, position
- Facing bet strategy with equity/pot odds analysis

---

### Task 4: Implement Session Logger ✅ COMPLETE
**Files Created:**
- `src/utils/session_logger.py`
- `database/learning/` directory

**Implemented:**
- SessionLogger class with log_decision() and log_hand_result()
- JSON Lines format for easy parsing
- Integrated into main.py game loop

---

### Task 5: Upgrade to CUDA PyTorch
**Estimated Complexity:** Low
**Files to Modify:**
- `requirements.txt`
- `test_installation.py`

**Instructions:**
1. Replace current PyTorch with CUDA version:
   ```
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
2. Update `requirements.txt` with correct CUDA versions
3. Add CUDA detection to `test_installation.py`:
   ```python
   import torch
   assert torch.cuda.is_available(), "CUDA not available"
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   ```

---

### Task 6: Complete Anchor Calibration Tool GUI
**Estimated Complexity:** Medium
**Files to Modify:**
- `src/ui/calibration_tool.py`
- `tools/calibrate_anchors.py`

**Instructions:**
1. Add GUI workflow:
   - Step 1: Select stable anchor point (e.g., dealer button, table edge)
   - Step 2: Draw rectangle around hole cards region
   - Step 3: Draw rectangle around community cards region
   - Step 4: Draw rectangle around pot display
   - Step 5: Draw rectangle around stack display
   - Step 6: Draw rectangle around bet display
2. Save all regions relative to anchor position
3. Add validation that all required regions are defined
4. Add preview mode showing regions overlaid on screenshot

---

### Task 7: Integrate PreflopStrategy into DecisionEngine ✅ COMPLETE
**Files Modified:**
- `src/strategy/decision_engine.py`

**Implemented:**
- PreflopStrategy integration with open/3bet/4bet action methods
- Automatic fallback to heuristics when GTO lookup fails
- Position-aware decision making

---

## Completed Medium Priority Tasks ✅

### Task 8: Add Board Texture Analysis ✅ COMPLETE
**Files Created:**
- `src/strategy/board_analyzer.py`

**Implemented:**
- BoardAnalyzer class with full texture analysis
- BoardTexture dataclass with all properties
- texture_score, connectivity, flush/straight detection

---

### Task 9: Add Pot Odds and EV Display to Overlay ✅ COMPLETE
**Files Modified:**
- `src/ui/display_manager.py`

**Implemented:**
- Pot odds display with required equity calculation
- Expected value (EV) display with color coding
- Green for +EV, red for -EV decisions

---

### Task 10: Create Performance Monitor ✅ COMPLETE
**Files Created:**
- `src/utils/performance.py`

**Implemented:**
- PerformanceMonitor class with context manager tracking
- Tracks: window_find, screen_capture, anchor_detection, card_detection, ocr_reading, decision_engine, total_frame
- Automatic warnings for slow operations
- Periodic summary logging every 100 frames

---

## Low Priority Tasks (V2 Prep)

### Task 11: Multi-table Foundation
- Create `TableManager` class that can track multiple windows
- Each table gets its own `GameLoop` instance
- Shared strategy engine

### Task 12: Hand History Export
- Export completed hands to standard hand history format
- Support PokerStars format for analysis tools

### Task 13: Opponent Modeling Foundation
- Track opponent actions over time
- Calculate VPIP/PFR/AF statistics
- Store in SQLite database

---

## Testing Requirements

For each task, ensure:
1. `python test_installation.py` passes
2. `python test_full_system.py` passes
3. No new warnings/errors in logs
4. Code follows existing patterns (type hints, docstrings, logger usage)

---

## File Structure Reference

```
project/
├── config/
│   ├── anchor_config.json    # Anchor + regions (needs Task 2 calibration)
│   └── settings.json         # App settings
├── database/
│   ├── preflop/              # ✅ Complete
│   │   ├── open_ranges.json
│   │   ├── 3bet_ranges.json
│   │   └── 4bet_ranges.json
│   ├── spots/                # Future: solver outputs
│   └── learning/             # ✅ Session logs stored here
├── src/
│   ├── strategy/
│   │   ├── preflop_strategy.py   # ✅ GTO preflop ranges
│   │   ├── postflop_strategy.py  # ✅ Texture-aware postflop
│   │   ├── decision_engine.py    # ✅ Integrated
│   │   └── board_analyzer.py     # ✅ Board texture analysis
│   └── utils/
│       ├── session_logger.py     # ✅ Decision logging
│       └── performance.py        # ✅ Frame timing
└── tools/
    ├── generate_gto_ranges.py    # ✅ Range generator
    └── list_windows.py           # ✅ Window listing utility
```

---

## Contact / Questions

If you encounter blockers or need clarification:
1. Check existing code patterns in similar modules
2. Review HANDOFF.md for context
3. Check test files for expected behavior
4. Leave TODO comments for unresolved questions
