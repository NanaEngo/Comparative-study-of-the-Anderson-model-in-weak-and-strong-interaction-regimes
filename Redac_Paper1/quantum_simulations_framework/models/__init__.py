"""
Models module for quantum agrivoltaic simulations.

This module contains model classes for analyzing molecular systems,
biodegradability, environmental factors, sensitivity analysis,
and testing validation protocols.
"""

try:
    from models.biodegradability_analyzer import BiodegradabilityAnalyzer
except ImportError:
    BiodegradabilityAnalyzer = None

try:
    from models.sensitivity_analyzer import SensitivityAnalyzer
except ImportError:
    SensitivityAnalyzer = None

try:
    from models.testing_validation_protocols import TestingValidationProtocols
except ImportError:
    TestingValidationProtocols = None

try:
    from models.lca_analyzer import LCAAnalyzer
except ImportError:
    LCAAnalyzer = None

__all__ = [
    'BiodegradabilityAnalyzer',
    'SensitivityAnalyzer',
    'TestingValidationProtocols',
    'LCAAnalyzer',
]
