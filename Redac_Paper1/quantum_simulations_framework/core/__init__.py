"""
Core module for quantum agrivoltaic simulations.

This module contains fundamental classes and constants for the simulation framework.
"""

from core.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_HIERARCHY,
    DEFAULT_TIME_POINTS,
    DEFAULT_REORGANIZATION_ENERGY,
    FMO_SITE_ENERGIES_7,
    FMO_COUPLINGS,
)

try:
    from core.hops_simulator import HopsSimulator
except ImportError:
    HopsSimulator = None

__all__ = [
    'HopsSimulator',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_MAX_HIERARCHY',
    'DEFAULT_TIME_POINTS',
    'DEFAULT_REORGANIZATION_ENERGY',
    'FMO_SITE_ENERGIES_7',
    'FMO_COUPLINGS',
]
