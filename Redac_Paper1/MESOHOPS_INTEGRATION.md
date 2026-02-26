# MesoHOPS Integration Guide

**Framework**: Quantum Agrivoltaic HOPS Research  
**Target Environment**: MesoHOPS  
**Date**: 2026-02-25

---

## Overview

The Quantum Agrivoltaic HOPS framework is **designed to run in the MesoHOPS environment**. This guide explains:
- How the framework integrates with MesoHOPS
- What version requirements exist
- How to verify API compatibility
- How to handle the fallback simulators

---

## MesoHOPS Integration Architecture

```
┌─────────────────────────────────────────────────────┐
│   Your Application / Research Code                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│   Quantum Agrivoltaic HOPS Framework                │
│   ├─ core/hops_simulator.py (Main integration)     │
│   ├─ models/ (Analysis & optimization)             │
│   └─ extensions/ (PT-HOPS, SBD enhancements)      │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼──────┐          ┌────────▼─────┐
    │ MesoHOPS  │          │  Fallback    │
    │ (Primary) │          │ Simulator    │
    │ When Avail│          │ (When N/A)   │
    └───────────┘          └──────────────┘
```

### Primary Path: MesoHOPS Environment
```python
HopsSimulator(use_mesohops=True)
├─ Initializes MesoHOPS system
├─ Creates basis and trajectory objects
├─ Runs accurate non-Markovian dynamics
└─ Leverages PT-HOPS and SBD optimizations
```

### Fallback Path: When MesoHOPS Unavailable
```python
HopsSimulator(use_mesohops=False)
├─ Falls back to SimpleQuantumDynamicsSimulator
├─ Runs basic quantum dynamics
└─ Reduced accuracy/features
```

---

## MesoHOPS Version Requirements

### Currently Supported
The framework targets **MesoHOPS version compatibility**.

**You must determine:**
1. Which MesoHOPS version is installed in your environment
2. Whether the API paths below match that version
3. Add version pinning to your requirements

### Critical API Paths to Verify

**Lines 179, 195** in `core/hops_simulator.py`:
```python
# These imports must be available in your MesoHOPS version:
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.trajectory.exp_noise import bcf_exp
```

**How to verify:**
```bash
# In Python with your MesoHOPS environment active:
python -c "from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp"
python -c "from mesohops.trajectory.exp_noise import bcf_exp"

# Both should import without errors
```

### Version Documentation Template

Create a file: `MESOHOPS_VERSION_REQUIREMENTS.md`

```markdown
# MesoHOPS Version Requirements

## Current Deployment
- **MesoHOPS Version**: [YOUR VERSION]
- **Tested With**: [Confirm these imports work]
- **Installation Method**: [e.g., pip install, conda, from source]

## API Compatibility
- ✓ `mesohops.util.bath_corr_functions.bcf_convert_dl_to_exp` - Available
- ✓ `mesohops.trajectory.exp_noise.bcf_exp` - Available

## Known Issues
[Document any version-specific quirks]

## Fallback Behavior
- When MesoHOPS unavailable: Framework uses SimpleQuantumDynamicsSimulator
- Accuracy: Reduced but functional
- Features: Basic quantum dynamics only
```

---

## Setup Instructions for MesoHOPS Environment

### 1. Create Requirements File

**File**: `requirements-mesohops.txt`
```txt
# MesoHOPS Environment - Framework Dependencies

# Primary MesoHOPS (set version based on your environment)
mesohops>=0.1.0,<1.0.0  # Adjust to match your version

# Required dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0

# Optional but recommended
pytest>=6.0          # For running framework tests
jupyter>=1.0         # For notebooks
notebook>=6.0        # For Jupyter notebooks
```

### 2. Update Framework Imports

**File**: `quantum_simulations_framework/core/hops_simulator.py`

Current (works in MesoHOPS env with PYTHONPATH):
```python
from utils.logging_config import get_logger
```

Better (works everywhere):
```python
from .utils.logging_config import get_logger
```

### 3. Document Version Compatibility

Add to `README.md`:
```markdown
## MesoHOPS Environment

This framework is designed to run in the MesoHOPS environment.

### Version Compatibility
- **MesoHOPS**: [Your version]
- **Python**: 3.8+
- **NumPy**: >=1.21.0
- **SciPy**: >=1.7.0

### Setup
1. Install/activate MesoHOPS environment
2. Install framework dependencies: `pip install -r requirements-mesohops.txt`
3. Verify: `python -c "from quantum_simulations_framework import HopsSimulator"`
```

### 4. Verify Installation

```python
# test_mesohops_setup.py
from quantum_simulations_framework.core import HopsSimulator
from quantum_simulations_framework.core.constants import FMO_SITE_ENERGIES_7
import numpy as np

# Create simple test
H = np.diag(FMO_SITE_ENERGIES_7)

# Initialize simulator (will use MesoHOPS if available)
try:
    sim = HopsSimulator(H, use_mesohops=True)
    print("✓ MesoHOPS integration working")
except ImportError:
    print("✓ Fallback simulator available")
except Exception as e:
    print(f"✗ Error: {e}")
```

Run:
```bash
python test_mesohops_setup.py
```

---

## Debugging MesoHOPS Integration

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'mesohops'`

**Solution**:
1. Verify MesoHOPS is installed: `pip list | grep mesohops`
2. Check PYTHONPATH includes MesoHOPS: `echo $PYTHONPATH`
3. If not found, install: `pip install mesohops`

### Issue: API Mismatch

**Error**: `ImportError: cannot import name 'bcf_convert_dl_to_exp'`

**Solution**:
1. Check MesoHOPS version: `python -c "import mesohops; print(mesohops.__version__)"`
2. Verify available functions:
   ```python
   import mesohops.util.bath_corr_functions as bcf
   print(dir(bcf))  # See what's available
   ```
3. Update imports if necessary
4. Document in version requirements

### Issue: Incorrect API Path

**Error**: `ImportError: cannot import name 'bcf_exp'`

**Solution**:
1. List available noise functions:
   ```python
   import mesohops.trajectory.exp_noise as en
   print(dir(en))
   ```
2. Find correct function name
3. Update `core/hops_simulator.py` line 195
4. Document in code comments

### Issue: Version Incompatibility

**Symptom**: Simulations produce incorrect results

**Solution**:
1. Check MesoHOPS changelog for breaking changes
2. Review API compatibility in your version
3. Add version compatibility tests:
   ```python
   def test_mesohops_version_compatibility():
       import mesohops
       required_version = (0, 1, 0)  # Example
       actual_version = tuple(map(int, mesohops.__version__.split('.')))
       assert actual_version >= required_version, \
           f"MesoHOPS {required_version}+ required, got {actual_version}"
   ```

---

## MesoHOPS-Specific Features Used

### 1. Process Tensor HOPS (PT-HOPS)
**Location**: `extensions/mesohops_adapters.py`  
**Purpose**: Efficient handling of non-Markovian effects  
**Requires**: MesoHOPS with PT-HOPS support

### 2. Stochastically Bundled Dissipators (SBD)
**Location**: `extensions/spectral_bundling.py`  
**Purpose**: Scalable simulation of large systems (>1000 chromophores)  
**Requires**: MesoHOPS with SBD implementation

### 3. HopsTrajectory Object
**Usage**: Core dynamics simulation  
**Documentation**: MesoHOPS documentation

### 4. Bath Correlation Functions
**Location**: Spectral density calculations  
**APIs**: `mesohops.util.bath_corr_functions`

---

## Integration Testing

### Test 1: Basic Import
```python
# Verify framework imports in MesoHOPS environment
from quantum_simulations_framework.core.hops_simulator import HopsSimulator
assert HopsSimulator is not None
```

### Test 2: MesoHOPS Availability
```python
# Check if MesoHOPS is available
try:
    from mesohops.basis.hops_system import HopsSystem
    print("✓ MesoHOPS available")
except ImportError:
    print("⚠ MesoHOPS not available, using fallback")
```

### Test 3: API Compatibility
```python
# Verify critical API paths
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.trajectory.exp_noise import bcf_exp
print("✓ All MesoHOPS APIs available")
```

### Test 4: Simulation
```python
# Run simple dynamics
import numpy as np
from quantum_simulations_framework.core import HopsSimulator
from quantum_simulations_framework.core.constants import FMO_SITE_ENERGIES_7

H = np.diag(FMO_SITE_ENERGIES_7)
sim = HopsSimulator(H, use_mesohops=True)
time_points = np.linspace(0, 100, 50)
results = sim.simulate_dynamics(time_points)
print(f"✓ Simulation completed: {len(results)} outputs")
```

---

## Best Practices for MesoHOPS Environment

### 1. Always Document Version
```python
# In your scripts/notebooks, add:
import mesohops
print(f"MesoHOPS version: {mesohops.__version__}")
```

### 2. Use Version Pinning
```txt
# requirements-mesohops.txt
mesohops==0.1.5  # Specific version that works with this framework
```

### 3. Validate Configuration
```python
def validate_mesohops_setup():
    """Verify MesoHOPS is correctly installed and compatible."""
    try:
        from mesohops.basis.hops_system import HopsSystem
        from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
        print("✓ MesoHOPS setup validated")
        return True
    except ImportError as e:
        print(f"✗ MesoHOPS setup issue: {e}")
        return False
```

### 4. Handle Fallback Gracefully
```python
sim = HopsSimulator(H, use_mesohops=True)
if not sim.use_mesohops:
    logger.warning("MesoHOPS not available, using fallback simulator")
```

### 5. Add Integration Tests
```bash
# Add to your test suite
pytest tests/test_mesohops_integration.py -v
```

---

## Troubleshooting Checklist

- [ ] MesoHOPS installed: `pip list | grep mesohops`
- [ ] Correct version: Check version requirements document
- [ ] PYTHONPATH set correctly: Framework can find utils module
- [ ] API paths verified: Both import statements work
- [ ] Test imports successful: Run test script above
- [ ] Simulation runs: Basic test completes without errors
- [ ] Results reasonable: Check population transfer patterns
- [ ] Documentation updated: Version requirements documented

---

## When to Use MesoHOPS vs Fallback

### Use MesoHOPS When:
- ✓ Full non-Markovian dynamics needed
- ✓ High accuracy required
- ✓ Large systems (>100 chromophores)
- ✓ Long simulation times (>500 fs)
- ✓ Research publication quality needed

### Fallback is OK For:
- ✓ Quick tests/prototyping
- ✓ Educational demonstrations
- ✓ Systems <50 chromophores
- ✓ Short simulations (<100 fs)
- ✓ MesoHOPS not available in environment

---

## References

- **MesoHOPS Documentation**: [Link to MesoHOPS docs]
- **Framework README**: `README.md`
- **API Reference**: `AGENTS.md`
- **Version Requirements**: Create `MESOHOPS_VERSION_REQUIREMENTS.md`

---

**Last Updated**: 2026-02-25  
**Framework Version**: 1.0.0  
**Status**: Designed for MesoHOPS Environment
