"""
Spectral Optimizer for Agrivoltaic Systems

This module implements multi-objective optimization algorithms to find
optimal spectral transmission functions for quantum-enhanced agrivoltaic systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eig, expm
from scipy.integrate import quad, trapezoid
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Import the other classes
from quantum_dynamics_simulator import QuantumDynamicsSimulator
from agrivoltaic_coupling_model import AgrivoltaicCouplingModel


class SpectralOptimizer:
    """
    Spectral optimizer implementing multi-objective optimization for agrivoltaic systems.
    
    This class implements advanced optimization algorithms to find optimal
    spectral transmission functions that simultaneously maximize OPV efficiency
    and photosynthetic efficiency in an agrivoltaic configuration.
    """
    
    def __init__(self, agrivoltaic_model, quantum_simulator=None, n_processes=None):
        """
        Initialize the spectral optimizer.
        
        Mathematical Framework:
        The spectral optimization problem is formulated as a multi-objective
        optimization:
        
        max_{T(λ)} [PCE(T), ETR(T), SPCE(T)]
        
        where PCE is the power conversion efficiency of OPV, ETR is the
        electron transport rate of photosynthetic units, and SPCE is the
        synergistic power conversion efficiency that captures cooperative effects.
        
        The optimization uses a weighted sum approach:
        
        max_T w₁*PCE(T) + w₂*ETR(T) + w₃*SPCE(T)
        
        subject to physical constraints on the transmission function T(λ).
        
        Parameters:
        agrivoltaic_model (AgrivoltaicCouplingModel): The coupling model
        quantum_simulator (QuantumDynamicsSimulator): Optional quantum simulator for enhanced metrics
        n_processes (int): Number of processes to use for parallel optimization (default: None for single process)
        """
        self.model = agrivoltaic_model
        self.quantum_simulator = quantum_simulator
        self.n_processes = n_processes  # Store number of processes for parallel optimization
        
        # Optimization bounds: for each parameter [min, max]
        # Each filter has 3 parameters: amplitude, center wavelength, width
        self.bounds = [(0.0, 1.0), (300, 1100), (5, 100)]  # For 1 filter
        # If multiple filters are used, bounds are repeated
        self.n_filters = 3  # Default: use 3 filters
        self.bounds = self.bounds * self.n_filters
    
    def objective_function(self, params):
        """
        Calculate the objective function for optimization.
        
        Mathematical Framework:
        The objective function combines multiple objectives with weights:
        
        f(params) = -(w₁*PCE + w₂*ETR + w₃*SPCE)
        
        The negative sign is used because we're minimizing (scipy.optimize.minimize
        is a minimizer, but we want to maximize efficiency).
        
        The weights can be adjusted to prioritize different objectives:
        - w₁: Weight for PCE (OPV efficiency)
        - w₂: Weight for ETR (photosynthetic efficiency)
        - w₃: Weight for SPCE (synergistic effects)
        
        Parameters:
        params (array): Filter parameters [A1, λ1, σ1, A2, λ2, σ2, ...]
        
        Returns:
        objective (float): Negative weighted sum of objectives (for minimization)
        """
        # Calculate transmission function from parameters
        try:
            T = self.model.calculate_spectral_transmission(params)
        except:
            # If calculation fails, return a large positive value (bad solution)
            return 100.0
        
        # Calculate individual objectives
        try:
            pce = self.calculate_pce(T)
        except:
            pce = 0.0
            
        try:
            etr = self.calculate_etr(T)
        except:
            etr = 0.0
            
        try:
            spce = self.calculate_synergistic_performance(T)
        except:
            spce = 0.0
        
        # Weights for the objectives (these can be adjusted)
        w_pce = 0.4  # Weight for PCE
        w_etr = 0.4  # Weight for ETR
        w_spce = 0.2  # Weight for synergistic performance
        
        # Calculate weighted objective (negative for minimization)
        objective = -(w_pce * pce + w_etr * etr + w_spce * spce)
        
        return objective

    def calculate_pce(self, transmission):
        """
        Calculate power conversion efficiency of the OPV system.
        
        Parameters:
        transmission (array or callable): Spectral transmission function
        
        Returns:
        pce (float): Power conversion efficiency
        """
        return self.model.calculate_opv_efficiency(transmission)

    def calculate_etr(self, transmission):
        """
        Calculate electron transport rate of the photosynthetic system.
        
        Parameters:
        transmission (array or callable): Spectral transmission function
        
        Returns:
        etr (float): Electron transport rate
        """
        return self.model.calculate_psu_efficiency(transmission)

    def calculate_synergistic_performance(self, transmission):
        """
        Calculate synergistic performance metric.
        
        Mathematical Framework:
        The synergistic performance captures cooperative effects between
        OPV and PSU that arise from quantum-coherent spectral splitting:
        
        SPCE = α*PCE + β*ETR + γ*Quantum_Synergy
        
        where the quantum synergy term accounts for non-classical correlations
        between the two systems that enhance overall performance.
        
        Parameters:
        transmission (array or callable): Spectral transmission function
        
        Returns:
        spce (float): Synergistic power conversion efficiency
        """
        # Calculate individual performances
        pce = self.calculate_pce(transmission)
        etr = self.calculate_etr(transmission)
        
        # Calculate quantum synergy (simplified model)
        # In a full implementation, this would involve quantum coherence calculations
        quantum_synergy = np.sqrt(pce * etr)  # Geometric mean as a simple synergy measure
        
        # Weighted combination
        alpha = 0.4
        beta = 0.4
        gamma = 0.2
        
        spce = alpha * pce + beta * etr + gamma * quantum_synergy
        
        return spce

    def optimize_spectral_splitting(self, method='differential_evolution', maxiter=15, popsize=15):
        """
        Perform multi-objective optimization for spectral splitting.
        
        Mathematical Framework:
        The optimization uses evolutionary algorithms to find optimal
        transmission functions that balance OPV and photosynthetic performance:
        
        1. Differential Evolution: Global optimization method that works
           well with non-convex, multi-modal objective functions
           
        2. Multi-start approach: Multiple random starting points to
           avoid local optima
           
        3. Parameter bounds: Physical constraints on filter properties
           (amplitudes, wavelengths, widths)
        
        Parameters:
        method (str): Optimization method ('differential_evolution', 'multistart')
        maxiter (int): Maximum number of iterations
        popsize (int): Population size for evolutionary algorithms
        
        Returns:
        result (dict): Optimization results with optimal parameters and performance
        """
        if method == 'differential_evolution':
            # Determine number of workers for parallel optimization
            if self.n_processes is not None:
                # Use the configured number of processes
                n_workers = self.n_processes
            else:
                # Default to single process to avoid potential multiprocessing issues
                n_workers = 1
            
            # Use differential evolution with enhanced parameters
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                maxiter=maxiter,
                popsize=popsize,
                # Use enhanced parameters for better convergence
                strategy='best1bin',  # Strategy for creating trial vectors
                mutation=(0.5, 1),    # Mutation parameters
                recombination=0.7,    # Recombination parameter
                seed=42,              # Reproducible results
                workers=n_workers,    # Use configured number of processes
                disp=False
            )
            
            # Extract optimal parameters
            optimal_params = result.x
            optimal_performance = -result.fun  # Convert back to maximization
            
            # Calculate individual metrics for the optimal solution
            T_optimal = self.model.calculate_spectral_transmission(optimal_params)
            pce_opt = self.calculate_pce(T_optimal)
            etr_opt = self.calculate_etr(T_optimal)
            spce_opt = self.calculate_synergistic_performance(T_optimal)
            
            return {
                'success': result.success,
                'optimal_params': optimal_params,
                'optimal_performance': optimal_performance,
                'pce': pce_opt,
                'etr': etr_opt,
                'spce': spce_opt,
                'transmission_function': T_optimal,
                'message': result.message
            }
        
        elif method == 'multistart':
            # Multi-start optimization with random initial points
            best_result = None
            best_performance = float('-inf')
            
            for i in range(5):  # 5 random starts
                # Generate random starting point within bounds
                x0 = [np.random.uniform(b[0], b[1]) for b in self.bounds]
                
                try:
                    result = minimize(
                        self.objective_function,
                        x0,
                        method='L-BFGS-B',
                        bounds=self.bounds,
                        options={'maxiter': maxiter*10}  # More iterations per start
                    )
                    
                    if -result.fun > best_performance:
                        best_performance = -result.fun
                        best_result = result
                except:
                    continue  # Skip if optimization fails
            
            if best_result is not None:
                optimal_params = best_result.x
                optimal_performance = -best_result.fun
                
                # Calculate individual metrics for the optimal solution
                T_optimal = self.model.calculate_spectral_transmission(optimal_params)
                pce_opt = self.calculate_pce(T_optimal)
                etr_opt = self.calculate_etr(T_optimal)
                spce_opt = self.calculate_synergistic_performance(T_optimal)
                
                return {
                    'success': best_result.success,
                    'optimal_params': optimal_params,
                    'optimal_performance': optimal_performance,
                    'pce': pce_opt,
                    'etr': etr_opt,
                    'spce': spce_opt,
                    'transmission_function': T_optimal,
                    'message': best_result.message
                }
            else:
                # Return failure result
                return {
                    'success': False,
                    'optimal_params': None,
                    'optimal_performance': 0.0,
                    'pce': 0.0,
                    'etr': 0.0,
                    'spce': 0.0,
                    'transmission_function': np.ones_like(self.model.lambda_range),
                    'message': 'Optimization failed'
                }
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def analyze_spectral_performance(self, params_list):
        """
        Analyze spectral performance across a range of parameters.
        
        Parameters:
        params_list (list): List of parameter sets to evaluate
        
        Returns:
        performance_data (DataFrame): Performance metrics for each parameter set
        """
        results = []
        
        for params in params_list:
            try:
                T = self.model.calculate_spectral_transmission(params)
                
                pce = self.calculate_pce(T)
                etr = self.calculate_etr(T)
                spce = self.calculate_synergistic_performance(T)
                
                # Store results
                results.append({
                    'params': params,
                    'pce': pce,
                    'etr': etr,
                    'spce': spce,
                    'objective': -(0.4*pce + 0.4*etr + 0.2*spce)  # For consistency with optimizer
                })
            except:
                # Skip failed evaluations
                continue
        
        return pd.DataFrame(results)

    def create_transmission_profile(self, n_points=100):
        """
        Create a comprehensive transmission profile analysis.
        
        Parameters:
        n_points (int): Number of points for parameter sweep
        
        Returns:
        transmission_profiles (list): List of transmission functions with performance metrics
        """
        # Define some representative transmission profiles
        profiles = []
        
        # Profile 1: High transmission (OPV-focused)
        params1 = [0.1, 600, 100] * self.n_filters  # Minimal filtering
        T1 = self.model.calculate_spectral_transmission(params1)
        pce1 = self.calculate_pce(T1)
        etr1 = self.calculate_etr(T1)
        profiles.append({
            'name': 'OPV-Optimized',
            'params': params1,
            'transmission': T1,
            'pce': pce1,
            'etr': etr1,
            'description': 'Maximizes OPV efficiency'
        })
        
        # Profile 2: High reflection (PSU-focused)
        params2 = [0.8, 550, 200] * self.n_filters  # Broad blocking
        T2 = self.model.calculate_spectral_transmission(params2)
        pce2 = self.calculate_pce(T2)
        etr2 = self.calculate_etr(T2)
        profiles.append({
            'name': 'PSU-Optimized',
            'params': params2,
            'transmission': T2,
            'pce': pce2,
            'etr': etr2,
            'description': 'Maximizes photosynthetic efficiency'
        })
        
        # Profile 3: Balanced (compromise solution)
        params3 = [0.5, 500, 50, 0.5, 650, 50, 0.3, 450, 30]  # Mixed filtering
        T3 = self.model.calculate_spectral_transmission(params3)
        pce3 = self.calculate_pce(T3)
        etr3 = self.calculate_etr(T3)
        profiles.append({
            'name': 'Balanced',
            'params': params3,
            'transmission': T3,
            'pce': pce3,
            'etr': etr3,
            'description': 'Balanced OPV-PSU performance'
        })
        
        return profiles