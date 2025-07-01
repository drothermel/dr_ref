# Infer-Types Reference

**⚠️ Note**: These conclusions are preliminary. Our understanding of error measurement and comparison methodologies is still evolving. The effectiveness numbers presented here are based on specific measurement approaches that may not reflect the full picture of cross-directory type checking interactions.

## Overview

`infer-types` is a utility from the `libcst` (LibCST) project that automatically adds type annotations to Python code using type inference. It leverages both static analysis and type information from `.pyi` stub files.

## Installation

```bash
pip install "libcst[infer]"
# or
uv add --dev "libcst[infer]"
```

## Basic Usage

```bash
# Infer types for a module
python -m libcst.tool infer-types module.py

# Infer types for a directory
python -m libcst.tool infer-types src/

# With specific options
python -m libcst.tool infer-types --aggressive src/
```

## How It Works

1. **Static Analysis**: Analyzes code structure and patterns
2. **Stub Files**: Uses `.pyi` files if available
3. **Type Propagation**: Infers types from usage patterns
4. **AST Transformation**: Modifies code using LibCST's preserving parser

## Key Features

### Type Sources

1. **Literal Analysis**:
   ```python
   # Before
   def get_count():
       return 42
   
   # After
   def get_count() -> int:
       return 42
   ```

2. **Stub File Integration**:
   - Reads from `.pyi` files in the same directory
   - Uses typeshed for standard library
   - Can consume custom stub files

3. **Usage-Based Inference**:
   ```python
   # If it sees: x = get_value(); x.split()
   # It might infer: def get_value() -> str:
   ```

## Options and Flags

### `--aggressive`
More aggressive inference, may be less accurate:
```bash
python -m libcst.tool infer-types --aggressive src/
```

### `--pyi-path`
Specify custom stub file locations:
```bash
python -m libcst.tool infer-types --pyi-path stubs/ src/
```

### `--show-errors`
Display inference errors and warnings:
```bash
python -m libcst.tool infer-types --show-errors src/
```

## Effectiveness Analysis

### Preliminary Results (Measurement Dependent)

In our dr_gen experiments:
```
Baseline: 1245 mypy errors
After infer-types: ~1200 errors (estimated)
Reduction: ~45 errors (3.6%)
```

**Critical Note**: These numbers are highly dependent on:
- Presence of `.pyi` stub files
- Code patterns in the project
- How errors are measured (combined vs. individual directory checks)

### Comparison with Autotyping

| Feature | Autotyping | Infer-Types |
|---------|------------|-------------|
| Literal returns | ✅ Yes | ✅ Yes |
| None returns | ✅ Yes (with flag) | ✅ Yes |
| Parameter types | ⚠️ Limited | ⚠️ Limited |
| Uses stub files | ❌ No | ✅ Yes |
| Usage analysis | ❌ No | ⚠️ Some |
| Generic types | ❌ No | ⚠️ Limited |

## Limitations Discovered

1. **Limited Effectiveness**: In practice, achieved less error reduction than autotyping
2. **Stub File Dependency**: Best results require comprehensive `.pyi` files
3. **Conservative by Default**: Won't add uncertain annotations
4. **No Cross-Module Analysis**: Limited to file-local inference

## When to Use Infer-Types

### Good Use Cases

1. **Projects with Stub Files**: If you have `.pyi` files, infer-types can leverage them
2. **After Manual Stubbing**: Generate stubs first, then apply
3. **Standard Library Heavy**: Benefits from typeshed integration

### When to Avoid

1. **No Stub Files**: Limited effectiveness without stubs
2. **Complex Generic Types**: Won't infer `List[Dict[str, Any]]`
3. **Cross-Module Dependencies**: Can't follow imports deeply

## Integration Workflow

### With Stub Generation
```bash
# 1. Generate stubs
stubgen -p mypackage -o stubs/

# 2. Apply inference using stubs
python -m libcst.tool infer-types --pyi-path stubs/ src/

# 3. Verify with mypy
mypy src/
```

### Combined Tool Approach
```bash
# 1. First pass: autotyping for safe changes
autotyping --safe --none-return src/

# 2. Second pass: infer-types for stub-based inference  
python -m libcst.tool infer-types src/

# 3. Third pass: MonkeyType for runtime types
monkeytype run -m pytest
monkeytype apply mymodule
```

## Common Issues

1. **Import Errors**: May fail if imports aren't resolvable
2. **Syntax Preservation**: Sometimes reformats code unexpectedly
3. **Partial Inference**: May add incomplete annotations
4. **Version Sensitivity**: Behavior varies across libcst versions

## Best Practices

1. **Generate Stubs First**: Use `stubgen` before running infer-types
2. **Review Changes**: Always review inferred types for accuracy
3. **Test Thoroughly**: Inferred types may be incorrect
4. **Measure Impact**: Check mypy errors before/after

## Example Commands

### Basic Inference
```bash
python -m libcst.tool infer-types src/
```

### With Custom Stubs
```bash
# Generate stubs
stubgen -p mypackage -o stubs/

# Use stubs for inference
python -m libcst.tool infer-types --pyi-path stubs/ src/
```

### Aggressive Mode
```bash
python -m libcst.tool infer-types --aggressive --show-errors src/
```

## Effectiveness Summary

Based on preliminary experiments:
- **Error Reduction**: ~3-5% of total mypy errors
- **Best Case**: Projects with existing stub files
- **Worst Case**: Projects with complex runtime types
- **Recommendation**: Use as a supplementary tool, not primary solution

## Conclusion

Infer-types provides modest improvements in type coverage, particularly when stub files are available. However, our experiments suggest it's less effective than autotyping for most codebases. Its real value may be in leveraging existing type information from stubs rather than inferring new types. As with all typing tools, measure its effectiveness using consistent methodology across all project directories to account for cross-directory error effects.