# Library Installation Status Report

## Date: 2025-01-01

### Summary of Required Libraries for Tier 4

Based on my investigation, here's the status of the libraries mentioned in the Tier 4 implementation plan:

## 1. Libraries Already in deconCNN pyproject.toml ✅

- **PyHessian** (`pyhessian>=0.1`) - Already added, available on PyPI
- **hessian-eigenthings** - Already added as git dependency from GitHub
- **scipy** (`>=1.11.0`) - Already present
- **click** (`>=8.2.1`) - Already present for CLI interfaces

## 2. Libraries Missing from Dependencies ❌

### Analysis & Visualization Libraries (from Tier 4 Step 1.1):
- **pandas** (`>=2.0.0`) - NOT present
- **seaborn** (`>=0.12.0`) - NOT present  
- **plotly** (`>=5.17.0`) - NOT present
- **jupyter** (`>=1.0.0`) - NOT present
- **matplotlib** (`>=3.7.0`) - NOT present

### Statistical Libraries:
- **statsmodels** - NOT present (mentioned in ground truth plan for OLS regression)

### Utility Libraries:
- **rich** - NOT present (for progress tracking and monitoring)

## 3. Libraries That Don't Exist as Standard Packages ⚠️

### hutch_nn
- **Status**: No standard Python package found
- **Findings**: Hutchinson trace estimation is available through:
  - BackPACK library (has Hutchinson trace estimation)
  - Custom implementations in research repositories
  - PyHessian likely includes Hutchinson trace functionality
- **Recommendation**: Use PyHessian's built-in Hutchinson trace or implement custom

## 4. Library Clarifications

### hessian-eigenthings
- **Installation**: Not on PyPI, must install from GitHub (already configured)
- **Command**: `pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings`
- **Purpose**: Efficient PyTorch Hessian eigendecomposition

### PyHessian
- **Installation**: Available on PyPI as `PyHessian`
- **Version**: 0.1 available
- **Purpose**: Second-order analysis of neural networks, includes Hutchinson trace estimation

## 5. Recommended Actions

### Immediate Additions Needed to pyproject.toml:
```toml
[project.dependencies]
# Add these to existing dependencies:
"pandas>=2.0.0",
"matplotlib>=3.7.0",
"statsmodels>=0.14.0",
"rich>=13.0.0",

[dependency-groups.analysis]
# Create new group for analysis-specific deps
"seaborn>=0.12.0",
"plotly>=5.17.0",
"jupyter>=1.0.0",
"notebook>=7.0.0",
```

### Implementation Strategy for Missing Functionality:

1. **Hutch++ Algorithm**: 
   - Since `hutch_nn` doesn't exist, implement Hutch++ using PyHessian's trace estimation
   - Or create custom implementation based on the algorithm paper

2. **Existing Alternatives**:
   - PyHessian provides Hutchinson trace estimation functionality
   - BackPACK could be an alternative for trace estimation
   - scipy.optimize can replace some statsmodels functionality

## 6. Installation Commands

To add the missing dependencies to deconCNN:
```bash
cd /path/to/deconCNN
uv add pandas matplotlib statsmodels rich
uv add --group analysis seaborn plotly jupyter notebook
```

## 7. Verification Steps

After installation, verify with:
```python
# Test imports
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
from rich.progress import track
import seaborn as sns
import plotly.graph_objects as go

# Test PyHessian
from pyhessian import hessian

# Test hessian-eigenthings  
from hessian_eigenthings import compute_hessian_eigenthings
```

## Conclusion

Most required libraries are standard and easily installable. The main gap is `hutch_nn`, which appears to be a non-existent package that should be replaced with PyHessian's functionality or a custom implementation. All visualization and analysis libraries need to be added to the deconCNN dependencies.