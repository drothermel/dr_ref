# Agent Implementation Prompt

## Your Task
You are tasked with implementing the next uncompleted commit from the optimized implementation plan for the H1/H2 hypothesis testing project. This involves adding monitoring capabilities to the deconCNN training framework to track power-law loss curves and early training dynamics.

## Context
- **Project Goal**: Test whether (H1) training loss follows a power law and (H2) early slope predicts final performance
- **Codebase**: deconCNN (PyTorch Lightning-based CNN training framework)
- **Location**: `~/drotherm/repos/deconCNN/`
- **Plan Location**: `~/drotherm/repos/dr_ref/projects/loss_lin_slope/impl_plans/tier1_2_optimized_implementation.md`

## Instructions

1. **Read the Implementation Plan**
   - Open and carefully read the optimized implementation plan
   - The plan contains 18 commits organized into 5 phases
   - Each commit has: code changes, commit message, and validation steps

2. **Identify Next Commit**
   - Check git log to see which commits have been completed
   - Find the next uncompleted commit in the sequence
   - Note its dependencies (some commits can be done in parallel)

3. **Execute the Commit**
   - Make the exact code changes specified
   - Run the validation steps
   - Use the provided commit message exactly
   - If validation fails, debug and fix before committing

4. **Testing Protocol**
   - After each commit, run its specific validation
   - For callbacks: test in isolation first
   - For integration: run the full test script
   - Ensure performance overhead stays under 15%

5. **Parallel Tracks**
   If working in a team, these tracks can progress independently:
   - **Track A**: Model changes (Commits 6-7)
   - **Track B**: Logging infrastructure (Commits 8-11)
   - **Track C**: Advanced callbacks (Commits 12-14)
   
   Note: All tracks depend on Phase 0 (Commits 1-5) being complete first.

## Key Files to Modify

- `pyproject.toml` - Dependencies
- `src/deconcnn/trainer.py` - Callback integration
- `src/deconcnn/module.py` - Logging changes
- `src/deconcnn/models/resnet.py` - Width multiplier
- `src/deconcnn/callbacks/` - New callback implementations
- `configs/` - Hydra configurations

## Critical Path (if time-limited)

If you need to deliver core functionality quickly, focus on these 5 commits only:
1. Dependencies + callback support (Commits 1-2)
2. Width multiplier (Commit 6)
3. Per-batch logging (Commit 8)
4. Gradient monitoring (Commit 10)
5. Integration test (Commit 18)

## Success Criteria

- All tests pass
- Performance overhead < 15%
- Logs contain all specified metrics
- Code follows existing patterns
- Commit messages match plan exactly

## Getting Started

```bash
cd ~/drotherm/repos/deconCNN
git status  # Check current state
git log --oneline -10  # See completed commits

# Read the plan
cat ~/drotherm/repos/dr_ref/projects/loss_lin_slope/impl_plans/tier1_2_optimized_implementation.md

# Start with next uncompleted commit
```

Remember: The plan is designed to be followed sequentially within each phase, but phases can sometimes be parallelized. Always check dependencies before starting a commit.