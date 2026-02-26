# Codebase Audit - Specific Fixes

This document provides concrete code fixes for issues identified in `CODEBASE_AUDIT.md`.

## Critical Issues - Immediate Fixes Needed

### Fix 1: Import Path Error in `core/hops_simulator.py`

**Line 46** - Change from absolute to relative import:

```python
# BEFORE (Line 46):
from utils.logging_config import get_logger

# AFTER:
from ..utils.logging_config import get_logger
# OR within package context:
from .utils.logging_config import get_logger
```

### Fix 2: MesoHOPS Import Issues

**Lines 179 & 195** - Verify correct MesoHOPS API:

```python
# BEFORE (Line 179):
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp

# AFTER - Check actual available function:
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp_with_Matsubara

# BEFORE (Line 195):  
from mesohops.trajectory.exp_noise import bcf_exp

# AFTER - May need to use:
from mesohops.trajectory.exp_noise import noise_exp_factory
# Or similar - verify with mesohops documentation
```

### Fix 3: Duplicate Imports

**Line 134** - Remove duplicate numpy import:

```python
# BEFORE:
def _setup_mesohops_if_available(self, time_points):
    import numpy as np  # DUPLICATE!
    
# AFTER:
def _setup_mesohops_if_available(self, time_points):
    # Use np from module level (line 10)
```

---

## High Priority Issues

### Fix 4: Replace Broad Exception Handling

Create a pattern to follow throughout the file:

```python
# PATTERN 1: Import errors
try:
    from mesohops.basis.hops_basis import HopsBasis
except ImportError as e:
    logger.warning(f"MesoHOPS HopsBasis not available: {e}")
    HopsBasis = None

# PATTERN 2: Configuration errors  
try:
    result = self.system.compute(params)
except (KeyError, ValueError) as e:
    logger.error(f"Invalid system configuration: {e}")
    raise RuntimeError(f"Failed to configure MesoHOPS system") from e

# PATTERN 3: Simulation errors
try:
    trajectory = HopsTrajectory(system, basis, integrator, exp_ops)
except (TypeError, RuntimeError) as e:
    logger.error(f"Failed to create trajectory: {e}")
    raise

# AVOID:
except Exception:  # TOO BROAD
    pass  # SILENT FAILURE

except Exception as e:  # CATCHES EVERYTHING
    logger.error(f"Error: {e}")  # Loses context
```

### Fix 5: Remove Unused Imports

```python
# BEFORE (Line 8):
import logging  # UNUSED - we use logger object

# BEFORE (Line 12):
import sys  # REIMPORTED on line 410

# BEFORE (Line 16):
from mesohops.basis.hops_system import HopsSystem  # UNUSED

# AFTER - Remove lines 8, 12, 16 and clean up line 410

# Clean line 410:
# Instead of: import sys
# Use module-level sys (but also unused, so remove)
```

### Fix 6: Protected Member Access

**Line 393** - Use public API:

```python
# BEFORE:
pt_noise._prepare_noise(self.system_param['L_NOISE1'], time_points=time_points)

# AFTER - Check if public method exists:
if hasattr(pt_noise, 'prepare_noise'):  # Public method
    pt_noise.prepare_noise(self.system_param['L_NOISE1'], time_points=time_points)
elif hasattr(pt_noise, '_prepare_noise'):  # Private method
    logger.warning("Using private API _prepare_noise - consider updating MesoHOPS")
    pt_noise._prepare_noise(self.system_param['L_NOISE1'], time_points=time_points)
```

### Fix 7: Unnecessary Pass Statement

**Line 381** - Remove or handle properly:

```python
# BEFORE:
except Exception:
    pass

# AFTER - Option 1: Remove if not needed
# Delete the try-except block

# AFTER - Option 2: Handle explicitly
except (AttributeError, TypeError) as e:
    logger.warning(f"Could not initialize PT-HOPS extensions: {e}")
    # Continue with fallback
```

### Fix 8: Exception Chaining

**Line 487** - Proper exception chaining:

```python
# BEFORE:
except Exception as e:
    raise RuntimeError(f"MesoHOPS simulation failed: {e}")

# AFTER:
except (RuntimeError, ValueError, KeyError) as e:
    raise RuntimeError(f"MesoHOPS simulation failed: {e}") from e
```

---

## Code Standardization

### Fix 9: Create requirements.txt

**File**: `/quantum_simulations_framework/requirements.txt`

```txt
# Core dependencies
numpy>=1.21.0,<2.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0

# Optional: Quantum dynamics (comment out if not needed)
mesohops>=0.1.0

# Development dependencies (optional)
pytest>=6.0
pytest-cov>=2.12
black>=21.0
pylint>=2.10
mypy>=0.910
```

### Fix 10: Add Configuration File

**File**: `/quantum_simulations_framework/pyproject.toml`

```toml
[project]
name = "quantum-agrivoltaics"
version = "1.0.0"
description = "Quantum simulations framework for agrivoltaic systems"
authors = [{name = "Research Team"}]
requires-python = ">=3.8"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.pylint]
max-line-length = 100
disable = ["C0103", "R0913"]  # Invalid-name, too-many-arguments

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

---

## Testing Improvements

### Fix 11: Expand Test Suite

**File**: `tests/test_core.py` - Add new test:

```python
def test_hops_simulator_relative_imports():
    """Test that HopsSimulator can be imported from package."""
    from quantum_simulations_framework.core.hops_simulator import HopsSimulator
    assert HopsSimulator is not None
    
def test_hops_simulator_exception_handling():
    """Test that HopsSimulator handles exceptions properly."""
    H = np.random.rand(7, 7)
    H = (H + H.T) / 2
    
    simulator = HopsSimulator(H, use_mesohops=False)
    
    # Test with invalid time points
    with pytest.raises((ValueError, TypeError)):
        simulator.simulate_dynamics(time_points=None)
        
def test_mesohops_availability():
    """Test MesoHOPS availability detection."""
    simulator = HopsSimulator(H, use_mesohops=True)
    # Should gracefully handle if MesoHOPS is not available
    assert simulator.use_mesohops in [True, False]
```

---

## Documentation Additions

### Fix 12: Create INSTALLATION.md

```markdown
# Installation Guide

## Requirements
- Python 3.8 or later
- pip or conda

## Basic Installation

```bash
cd quantum_simulations_framework
pip install -r requirements.txt
```

## Optional: MesoHOPS Installation

For full functionality with MesoHOPS:

```bash
pip install mesohops
```

If MesoHOPS is not available, the framework will use fallback simulators.

## Verification

```python
import quantum_simulations_framework as qaf
print(f"Version: {qaf.__version__}")
```

## Troubleshooting

- **ImportError on utils**: Ensure you're running from the correct directory
- **MesoHOPS not found**: Install with `pip install mesohops`
- **Version conflicts**: Run `pip install -r requirements.txt --upgrade`
```

### Fix 13: Create API.md

Document the main entry points:

```markdown
# API Reference

## Core Modules

### HopsSimulator
```python
from quantum_simulations_framework.core import HopsSimulator
simulator = HopsSimulator(hamiltonian, temperature=295)
results = simulator.simulate_dynamics(time_points)
```

### Environmental Factors  
```python
from quantum_simulations_framework.models import EnvironmentalFactors
env = EnvironmentalFactors()
pce, etr, dust = env.combined_environmental_effects(...)
```

### LCA Analysis
```python
from quantum_simulations_framework.models import LCAAnalyzer
lca = LCAAnalyzer()
results = lca.calculate_lca_impact(manufacturing_energy=1500)
```

[Continue for all major classes...]
```

---

## Implementation Priority

### Phase 1 (Immediate - 2-4 hours)
```
1. Fix import in core/hops_simulator.py line 46
2. Verify MesoHOPS API and fix lines 179, 195
3. Remove unused imports (8, 12, 16)
4. Remove duplicate import line 134
5. Create requirements.txt
```

### Phase 2 (Next - 4-8 hours)
```
1. Replace all broad exception handling
2. Remove/fix unnecessary pass statements
3. Add pyproject.toml
4. Create INSTALLATION.md
5. Run code formatter (black)
```

### Phase 3 (Follow-up - 1-2 days)
```
1. Expand test suite
2. Add API documentation
3. Set up CI/CD linting
4. Create example notebooks
```

---

## Verification Checklist

After applying fixes:

```bash
# Check imports work
python -c "from quantum_simulations_framework import HopsSimulator; print('✓ Imports OK')"

# Run tests
pytest tests/ -v

# Check code style
black --check quantum_simulations_framework/

# Check for type issues
mypy quantum_simulations_framework/ --ignore-missing-imports

# Check for linting issues
pylint quantum_simulations_framework/ --disable=C0103,R0913
```

---

**Generated**: 2026-02-25  
**Related Document**: CODEBASE_AUDIT.md
