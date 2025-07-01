# Autotyping Reference

**⚠️ Note**: These conclusions are preliminary. Our understanding of error measurement and comparison methodologies is still evolving, particularly regarding cross-directory error counting and context-dependent type checking.

## Overview

Autotyping is a tool that automatically adds type annotations to Python code using static analysis. It focuses on "safe" transformations that are guaranteed to be correct based on the code structure.

## Installation

```bash
pip install autotyping
# or
uv add --dev autotyping
```

## Basic Usage

```bash
# Add type annotations to a single file
autotyping module.py

# Add annotations to entire directory
autotyping src/

# Add annotations to multiple directories
autotyping src/ tests/ scripts/

# Safe mode (recommended)
autotyping --safe src/

# Add None return types
autotyping --none-return src/
```

## Key Options

### `--safe` (Recommended)
Only applies transformations that are guaranteed to be correct:
- Functions that return literals
- Functions with no return statements (adds `-> None`)
- Simple, unambiguous cases

### `--none-return`
Adds `-> None` to functions without return statements:
```python
# Before
def setup():
    self.initialized = True

# After  
def setup() -> None:
    self.initialized = True
```

### `--int-param`, `--str-param`, `--float-param`, `--bool-param`
Infer parameter types from default values:
```python
# Before
def process(count=0, name="", ratio=1.0, active=True):
    pass

# After (with all param flags)
def process(count: int = 0, name: str = "", ratio: float = 1.0, active: bool = True) -> None:
    pass
```

## What Autotyping Can Do

1. **Return Type Inference**:
   - Literal returns: `return 42` → `-> int`
   - None returns: no return → `-> None`
   - Simple expressions: `return x + y` (if types are known)

2. **Parameter Type Inference** (with flags):
   - From defaults: `def f(x=0):` → `def f(x: int = 0):`
   - From usage patterns (limited)

3. **Safe Transformations**:
   - Unambiguous cases only
   - Preserves existing annotations
   - Won't break existing code

## What Autotyping Cannot Do

1. **Complex Type Inference**:
   - Generic types: `List[str]`, `Dict[str, int]`
   - Union types: `str | int`
   - Optional types: `Optional[str]`
   - Custom classes (unless obvious)

2. **Cross-Function Analysis**:
   - Parameter types from call sites
   - Return types from usage
   - Type propagation across modules

3. **Runtime-Dependent Types**:
   - Types that depend on runtime values
   - Dynamic attribute access
   - Conditional returns with different types

## Effectiveness Analysis

### Preliminary Results (Subject to Measurement Methodology)

Based on our experiments with the dr_gen codebase:
```
Baseline: 1245 mypy errors
After autotyping --safe --none-return: 1075 errors
Reduction: 170 errors (13.7%)
```

**Important**: These numbers depend heavily on:
- How errors are counted (individual vs. combined directory checks)
- Cross-directory error effects
- The specific error types in your codebase

### Error Types Addressed

Autotyping primarily reduces:
- `no-untyped-def`: Functions missing type annotations
- Some `no-any-return`: Functions returning Any

### Limitations Discovered

1. **The "Automation Ceiling"**: In our tests, autotyping could only reduce errors by ~87-170 (depending on measurement method), leaving 80-90% of errors unresolved.

2. **Directory Scope Matters**: 
   ```bash
   autotyping --safe --none-return src/        # -87 errors
   autotyping --safe --none-return src/ tests/ scripts/  # -170 errors
   ```

3. **Doesn't Address**:
   - `var-annotated`: Missing variable annotations
   - `arg-type`: Argument type mismatches
   - Complex generic types

## Best Practices

1. **Always Use Safe Mode**: `--safe` prevents incorrect annotations
2. **Include All Directories**: Run on `src/`, `tests/`, and `scripts/` together
3. **Add None Returns**: Use `--none-return` for easy wins
4. **Run After**: Use as a first pass before manual annotation or MonkeyType
5. **Verify Results**: Always run mypy after to ensure error reduction

## Integration with Other Tools

### Before MonkeyType
```bash
# 1. First pass with autotyping
autotyping --safe --none-return src/ tests/ scripts/

# 2. Then use MonkeyType for runtime inference
monkeytype run -m pytest
monkeytype apply module
```

### With pyupgrade
```bash
# First modernize syntax
pyupgrade --py38-plus src/**/*.py

# Then add types
autotyping --safe --none-return src/
```

## Command Examples

### Conservative Approach
```bash
autotyping --safe --none-return src/
```

### Aggressive Approach  
```bash
autotyping --safe --none-return --int-param --str-param --bool-param src/ tests/
```

### Check Impact
```bash
# Before
mypy src/ tests/ scripts/ | grep -c "error:"

# Apply autotyping
autotyping --safe --none-return src/ tests/ scripts/

# After
mypy src/ tests/ scripts/ | grep -c "error:"
```

## Common Pitfalls

1. **Overconfidence**: Not all transformations are safe, even with `--safe`
2. **Incomplete Coverage**: Missing test/script directories reduces effectiveness
3. **Order Matters**: Run on clean code, before other type tools
4. **Not a Silver Bullet**: Expect to still have 80%+ of errors remaining

## Summary

Autotyping is a useful first-pass tool that can eliminate 10-15% of type errors through safe, static transformations. It works best as part of a multi-tool approach, providing a foundation for more sophisticated tools like MonkeyType to build upon. Always measure its effectiveness using consistent error counting methodology across all project directories.