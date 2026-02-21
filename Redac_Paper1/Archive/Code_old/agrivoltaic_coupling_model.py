"""
Agrivoltaic Coupling Model

This module implements the agrivoltaic coupling model that combines
organic photovoltaics (OPV) with photosynthetic units (PSU) to create
a quantum-enhanced agrivoltaic system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eig, expm
from scipy.integrate import quad, trapezoid


class AgrivoltaicCouplingModel:
    """
    Agrivoltaic coupling model combining OPV and photosynthetic systems.
    
    This class models the coupling between organic photovoltaic (OPV) systems
    and photosynthetic units (PSU) in an agrivoltaic configuration, allowing
    for optimal spectral utilization of solar irradiance.
    
    The model implements quantum-coherent spectral splitting where different
    spectral regions are preferentially absorbed by OPV or PSU components.
    """
    
    def __init__(self, fmo_hamiltonian, temperature=295, solar_spectrum=None, 
                 opv_bandgap=1.4, opv_absorption_coeff=1.0, n_opv_sites=4):
        """
        Initialize the agrivoltaic coupling model.
        
        Mathematical Framework:
        The agrivoltaic coupling model combines two energy harvesting systems:
        
        1. Organic Photovoltaic (OPV) system: Harvests high-energy photons
           with power conversion efficiency (PCE) dependent on absorbed spectrum
           
        2. Photosynthetic Unit (PSU): Harvests remaining photons optimized
           for photosynthetic efficiency, measured as Electron Transport Rate (ETR)
        
        The total system performance is evaluated using a spectral transmission
        function T(λ) that determines which photons go to OPV vs PSU:
        
        I_OPV(λ) = I_sun(λ) * T(λ)        (transmitted to OPV)
        I_PSU(λ) = I_sun(λ) * [1 - T(λ)]  (absorbed by PSU)
        
        Parameters:
        -----------
        fmo_hamiltonian : 2D array
            FMO Hamiltonian matrix
        temperature : float, optional
            Temperature in Kelvin, default is 295
        solar_spectrum : tuple, optional
            Wavelength array and irradiance array
        opv_bandgap : float, optional
            OPV bandgap in eV, default is 1.4
        opv_absorption_coeff : float, optional
            OPV absorption coefficient, default is 1.0
        n_opv_sites : int, optional
            Number of OPV sites for Hamiltonian model (default: 4)
        """
        self.fmo_hamiltonian = fmo_hamiltonian
        self.temperature = temperature
        
        # Hamiltonian Model Parameters
        self.n_opv_sites = n_opv_sites
        self.n_psu_sites = len(fmo_hamiltonian)
        self.n_total = self.n_opv_sites * self.n_psu_sites
        
        # Default parameters for OPV Hamiltonian (from refined model)
        self.opv_params = {
            'site_energies': np.array([1.8, 1.75, 1.85, 1.7]),  # eV
            'coupling_matrix': np.array([
                [0.0,  0.1,  0.05, 0.02],
                [0.1,  0.0,  0.08, 0.03],
                [0.05, 0.08, 0.0,  0.1],
                [0.02, 0.03, 0.1,  0.0]
            ]),
            'temperature': temperature
        }
        
        # Convert FMO Hamiltonian from cm^-1 to eV for Dynamics
        fmo_ham_eV = fmo_hamiltonian / 8065.54
        fmo_energies_eV = np.diag(fmo_ham_eV)
        
        self.psu_params = {
            'site_energies': fmo_energies_eV,  # Use FMO energies in eV
            'coupling_matrix': fmo_ham_eV - np.diag(fmo_energies_eV),  # Use FMO couplings (off-diagonal)
            'temperature': temperature
        }
        
        # Create Hamiltonians
        self.H_opv = self._create_opv_hamiltonian()
        self.H_psu = self._create_psu_hamiltonian()
        self.H_total = self._construct_agrivoltaic_hamiltonian()
        
        # Solar spectrum (default: AM1.5G)
        if solar_spectrum is None:
            # Create standard AM1.5G spectrum in the range 300-1100 nm
            lambda_range = np.linspace(300, 1100, 801)  # nm
            # Simplified AM1.5G spectrum (mW/cm²/nm)
            # This is a simplified approximation of the real spectrum
            solar_irradiance = np.zeros_like(lambda_range, dtype=float)
            
            # Fill with approximated AM1.5G values
            for i, lam in enumerate(lambda_range):
                if 300 <= lam <= 400:  # UV-Violet
                    solar_irradiance[i] = 0.5 + 1.2 * (lam - 300) / 100
                elif 400 <= lam <= 700:  # Visible
                    solar_irradiance[i] = 1.7 - 0.3 * abs(lam - 550) / 150
                elif 700 <= lam <= 1100:  # NIR
                    solar_irradiance[i] = 1.4 * np.exp(-0.002 * (lam - 700))
                else:
                    solar_irradiance[i] = 0.0
            
            # Normalize to standard irradiance (1000 W/m² = 100 mW/cm²)
            integral = trapezoid(solar_irradiance, lambda_range)
            solar_irradiance = solar_irradiance * 100.0 / integral  # mW/cm²/nm            
            self.lambda_range = lambda_range
            self.solar_spec = solar_irradiance
        else:
            self.lambda_range, self.solar_spec = solar_spectrum
        
        # OPV parameters
        self.opv_bandgap = opv_bandgap  # eV
        self.opv_absorption_coeff = opv_absorption_coeff
        
        # Calculate photosynthetic action spectrum (simplified)
        self.R_opv = self._calculate_opv_response()
        self.R_psu = self._calculate_psu_response()
    
    def _calculate_opv_response(self):
        """
        Calculate the OPV spectral response function.
        
        Mathematical Framework:
        The OPV response function R_OPV(λ) represents the quantum efficiency
        of the OPV device as a function of wavelength. It's determined by:
        
        R_OPV(λ) = η_internal(λ) * T(λ) * α(λ)
        
        where η_internal is the internal quantum efficiency, T is the transmission
        through optical layers, and α is the absorption coefficient.
        
        For wavelengths shorter than the bandgap (λ < λ_gap = 1240/E_gap in eV),
        the response is significantly reduced due to thermalization losses.
        
        Returns:
        --------
        R_opv : array
            OPV spectral response
        """
        # Convert bandgap to wavelength (nm): λ (nm) = 1240 / E_gap (eV)
        lambda_cutoff = 1240.0 / self.opv_bandgap
        
        # Simplified response function: high efficiency above bandgap, low below
        # Also account for typical OPV absorption characteristics
        R_opv = np.zeros_like(self.lambda_range, dtype=float)
        
        for i, lam in enumerate(self.lambda_range):
            if lam > lambda_cutoff:
                # Above bandgap: response decreases with wavelength
                # Typical OPV response curve
                R_opv[i] = self.opv_absorption_coeff * np.exp(-(lam - lambda_cutoff) / 100)
            else:
                # Below bandgap: very low response due to thermalization
                R_opv[i] = self.opv_absorption_coeff * 0.05 * (lam / lambda_cutoff)
        
        # Normalize to maximum of 1
        if np.max(R_opv) > 0:
            R_opv = R_opv / np.max(R_opv)
        
        return R_opv
    
    def _calculate_psu_response(self):
        """
        Calculate the photosynthetic unit spectral response function.
        
        Mathematical Framework:
        The PSU response function R_PSU(λ) represents the photosynthetic
        quantum efficiency as a function of wavelength. It's based on:
        
        1. Chlorophyll absorption spectra
        2. Photosystem II and I quantum yields
        3. Light-harvesting complex efficiency
        
        The response peaks in the blue (400-500 nm) and red (600-700 nm)
        regions where chlorophylls have strong absorption, with a valley
        in the green region (500-600 nm) where chlorophylls are less efficient.
        
        Returns:
        --------
        R_psu : array
            PSU spectral response
        """
        R_psu = np.zeros_like(self.lambda_range, dtype=float)
        
        for i, lam in enumerate(self.lambda_range):
            if 400 <= lam <= 500:  # Blue region
                # Chlorophyll absorption peak in blue
                R_psu[i] = 0.8 + 0.2 * np.sin(np.pi * (lam - 400) / 100)
            elif 600 <= lam <= 700:  # Red region
                # Chlorophyll absorption peak in red
                R_psu[i] = 0.85 + 0.15 * np.cos(np.pi * (lam - 650) / 50)
            elif 500 < lam < 600:  # Green valley
                # Lower efficiency in green region
                R_psu[i] = 0.2 + 0.1 * np.sin(np.pi * (lam - 500) / 100)
            elif lam < 400:  # UV region
                # Low efficiency due to photodamage protection
                R_psu[i] = 0.1
            else:  # Beyond 700 nm
                # Decreasing efficiency in NIR
                R_psu[i] = 0.3 * np.exp(-0.01 * (lam - 700))
        
        # Normalize to maximum of 1
        if np.max(R_psu) > 0:
            R_psu = R_psu / np.max(R_psu)
        
        return R_psu

    def calculate_opv_current(self, transmission_func):
        """
        Calculate the OPV current generated for a given transmission function.
        
        Mathematical Framework:
        The short-circuit current for the OPV is calculated as:
        
        J_sc = q ∫ I_sun(λ) * T(λ) * R_OPV(λ) dλ
        
        where q is the elementary charge (1.602e-19 C), I_sun is the solar
        irradiance, T is the transmission function, and R_OPV is the OPV
        spectral response.
        
        This current determines the power output of the OPV system.
        
        Parameters:
        -----------
        transmission_func : callable or array
            Spectral transmission function
        
        Returns:
        --------
        current : float
            OPV current in mA/cm²
        """
        # Convert transmission to array if it's a function
        if callable(transmission_func):
            T = transmission_func(self.lambda_range)
        else:
            T = transmission_func
        
        # Calculate absorbed irradiance for OPV
        absorbed_irradiance = self.solar_spec * T * self.R_opv
        
        # Integrate to get current (proportional to photon flux)
        current = trapezoid(absorbed_irradiance, self.lambda_range)
        
        # Convert from mW/cm²/nm to mA/cm² (simplified conversion)
        # Assuming an average photon energy for the spectrum
        current_mA_per_cm2 = current * 0.1  # Simplified conversion factor
        
        return current_mA_per_cm2

    def calculate_opv_efficiency(self, transmission_func, voltage=0.8):
        """
        Calculate the OPV power conversion efficiency (PCE).
        
        Mathematical Framework:
        The power conversion efficiency is given by:
        
        PCE = (V_oc * J_sc * FF) / P_in
        
        where V_oc is the open-circuit voltage (set to fixed value),
        J_sc is the short-circuit current calculated from the spectral
        absorption, FF is the fill factor (assumed constant), and P_in
        is the input power density.
        
        For the agrivoltaic system, the PCE calculation accounts for
        the fraction of spectrum transmitted to the OPV.
        
        Parameters:
        -----------
        transmission_func : callable or array
            Spectral transmission function
        voltage : float, optional
            Operating voltage in V (default: 0.8 for OPV)
        
        Returns:
        --------
        pce : float
            Power conversion efficiency
        """
        # Calculate short circuit current
        j_sc = self.calculate_opv_current(transmission_func)
        
        # Define fill factor and input power
        fill_factor = 0.7  # Typical for OPV
        input_power = trapezoid(self.solar_spec, self.lambda_range) / 1000.0  # W/m² to normalize
        
        # Calculate efficiency
        pce = (voltage * j_sc * fill_factor) / (input_power * 1000.0)  # Convert to proper units
        
        # Ensure non-negative efficiency
        return max(0.0, pce)

    def calculate_psu_efficiency(self, transmission_func):
        """
        Calculate the photosynthetic efficiency for the PSU.
        
        Mathematical Framework:
        The photosynthetic efficiency is calculated based on the absorbed
        spectrum and the FMO complex efficiency. The Electron Transport Rate (ETR)
        is calculated as:
        
        ETR = Φ_PSII * PAR_absorbed
        
        where Φ_PSII is the PSII quantum yield and PAR_absorbed is the
        photosynthetically active radiation absorbed by the PSU.
        
        The absorbed PAR is calculated as:
        
        PAR_absorbed = ∫ I_sun(λ) * [1 - T(λ)] * R_PSU(λ) dλ
        
        Parameters:
        -----------
        transmission_func : callable or array
            Spectral transmission function
        
        Returns:
        --------
        etr : float
            Electron Transport Rate (relative units)
        """
        # Convert transmission to array if it's a function
        if callable(transmission_func):
            T = transmission_func(self.lambda_range)
        else:
            T = transmission_func
        
        # Calculate absorbed irradiance for PSU
        absorbed_irradiance = self.solar_spec * (1 - T) * self.R_psu
        
        # Integrate to get absorbed photosynthetically active radiation
        par_absorbed = trapezoid(absorbed_irradiance, self.lambda_range)
        
        # Calculate ETR (simplified model)
        # Assume quantum yield is proportional to absorbed radiation
        # but saturates at high light intensities
        etr = par_absorbed / (1.0 + par_absorbed / 50.0)  # Saturation model
        
        return etr

    def calculate_spectral_transmission(self, params):
        """
        Calculate spectral transmission function based on filter parameters.
        
        Mathematical Framework:
        The spectral transmission is modeled as a superposition of Gaussian
        filters designed to optimize spectral splitting between OPV and PSU:
        
        T(λ) = 1 - Σ Aᵢ * exp(-(λ - λᵢ)² / (2σᵢ²))
        
        where Aᵢ, λᵢ, σᵢ are the amplitude, center wavelength, and width
        of each Gaussian filter. The filters are designed to transmit
        certain wavelengths to the OPV while reflecting others to the PSU.
        
        Parameters:
        -----------
        params : array
            Array of filter parameters [A1, λ1, σ1, A2, λ2, σ2, ...]
        
        Returns:
        --------
        transmission : array
            Spectral transmission values
        """
        n_filters = len(params) // 3
        if n_filters == 0:
            # Return neutral transmission (all light to OPV)
            return np.ones_like(self.lambda_range)
        
        # Extract parameters
        transmission = np.ones(len(self.lambda_range))
        
        for i in range(n_filters):
            A = params[3*i]      # Amplitude (0-1)
            center = params[3*i+1]  # Center wavelength (nm)
            width = params[3*i+2]   # Width (nm)
            
            # Ensure physical bounds
            A = np.clip(A, 0.0, 1.0)
            width = max(width, 1.0)  # Minimum width
            
            # Add Gaussian filter contribution
            gaussian_filter = A * np.exp(-(self.lambda_range - center)**2 / (2 * width**2))
            # Multiple filters multiply transmission (more blocking)
            transmission = transmission * (1 - gaussian_filter)
        
        # Ensure transmission is between 0 and 1
        transmission = np.clip(transmission, 0.0, 1.0)
        
        return transmission

    def create_filter_transmission(self, center_wavelength, fwhm, amplitude=1.0, shape='gaussian'):
        """
        Create a transmission function for a single filter.
        
        Mathematical Framework:
        Different filter shapes can be used to create spectral transmission
        profiles:
        
        Gaussian: T(λ) = 1 - A * exp(-(λ - λ₀)² / (2σ²))
        Lorentzian: T(λ) = 1 - A / (1 + ((λ - λ₀) / γ)²)
        
        where A is the amplitude, λ₀ is the center wavelength, σ is the
        standard deviation (σ = FWHM / (2√(2ln2))), and γ is the half-width
        at half-maximum for Lorentzian.
        
        Parameters:
        -----------
        center_wavelength : float
            Center wavelength in nm
        fwhm : float
            Full width at half maximum in nm
        amplitude : float, optional
            Filter amplitude (0-1), default is 1.0
        shape : str, optional
            Filter shape ('gaussian', 'lorentzian', 'tophat'), default is 'gaussian'
        
        Returns:
        --------
        transmission_func : callable
            Function returning transmission values
        """
        if shape == 'gaussian':
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            
            def transmission_func(wavelengths):
                if np.isscalar(wavelengths):
                    wavelengths = np.array([wavelengths])
                
                T = 1 - amplitude * np.exp(-(wavelengths - center_wavelength)**2 / (2 * sigma**2))
                return np.clip(T, 0.0, 1.0)
                
        elif shape == 'lorentzian':
            gamma = fwhm / 2.0  # Half width at half maximum
            
            def transmission_func(wavelengths):
                if np.isscalar(wavelengths):
                    wavelengths = np.array([wavelengths])
                
                T = 1 - amplitude / (1 + ((wavelengths - center_wavelength) / gamma)**2)
                return np.clip(T, 0.0, 1.0)
                
        elif shape == 'tophat':
            # Create a transmission function that blocks a specific range
            def transmission_func(wavelengths):
                if np.isscalar(wavelengths):
                    wavelengths = np.array([wavelengths])
                
                T = np.ones_like(wavelengths, dtype=float)
                mask = (wavelengths >= center_wavelength - fwhm/2) & (wavelengths <= center_wavelength + fwhm/2)
                T[mask] = 1 - amplitude
        return transmission_func

    # -------------------------------------------------------------------------
    # Hamiltonian Dynamics Methods (Ported from Refined Framework)
    # -------------------------------------------------------------------------

    def _create_opv_hamiltonian(self):
        """Create OPV subsystem Hamiltonian."""
        H = np.diag(self.opv_params['site_energies']) + self.opv_params['coupling_matrix']
        return H
    
    def _create_psu_hamiltonian(self):
        """Create PSU subsystem Hamiltonian based on FMO complex."""
        H = np.diag(self.psu_params['site_energies']) + self.psu_params['coupling_matrix']
        return H
    
    def _construct_agrivoltaic_hamiltonian(self, spectral_coupling_strength=0.05):
        """
        Construct the full agrivoltaic Hamiltonian using tensor products.
        
        Returns
        -------
        H_agri : 2D array
            Full agrivoltaic Hamiltonian
        """
        n_opv = self.n_opv_sites
        n_psu = self.n_psu_sites
        
        # Identity matrices
        I_opv = np.eye(n_opv)
        I_psu = np.eye(n_psu)
        
        # Tensor products for uncoupled terms
        H_opv_full = np.kron(self.H_opv, I_psu)
        H_psu_full = np.kron(I_opv, self.H_psu)
        
        # Spectral coupling term (simplified)
        coupling_matrix = np.zeros((n_opv * n_psu, n_opv * n_psu))
        # Connect ground states of each system
        opv_gs = 0  # OPV ground state index
        psu_gs = 0  # PSU ground state index
        overall_gs_idx = opv_gs * n_psu + psu_gs
        # Connect to first excited states
        overall_exc_idx = 1 * n_psu + 1  # Example excited state
        
        if overall_exc_idx < n_opv * n_psu:
            coupling_matrix[overall_gs_idx, overall_exc_idx] = spectral_coupling_strength
            coupling_matrix[overall_exc_idx, overall_gs_idx] = spectral_coupling_strength
        
        # Combine all terms
        H_agri = H_opv_full + H_psu_full + coupling_matrix
        
        return H_agri
    
    def calculate_opv_transmission(self, omega, peak_pos=1.8, peak_width=0.2, max_trans=0.7):
        """
        Calculate OPV transmission as function of frequency.
        """
        lorentzian = 1.0 / (1 + ((omega - peak_pos) / peak_width)**2)
        transmission = max_trans * (1 - lorentzian)  # High transmission outside absorption band
        return np.clip(transmission, 0, 1)
    
    def calculate_psu_absorption_from_hamiltonian(self, omega):
        """
        Calculate PSU absorption cross-section based on FMO complex Hamiltonian.
        """
        eigenvals = np.sort(np.real(np.linalg.eigvalsh(self.H_psu)))
        
        sigma = np.zeros_like(omega)
        broadening = 0.05  # eV, homogeneous broadening
        
        for eig in eigenvals:
            lorentzian = broadening / (np.pi * ((omega - eig)**2 + broadening**2))
            sigma += 0.5 * lorentzian  # oscillator strength
        
        # Normalize
        if np.max(sigma) > 0:
            sigma = sigma / np.max(sigma)
        
        return sigma
    
    def calculate_quantum_transmission_operator(self, omega, T_opv_values, PSU_cross_section):
        """
        Calculate quantum transmission operator T_quant(ω).
        """
        T_quant = T_opv_values * PSU_cross_section  # Element-wise multiplication
        return T_quant
    
    def simulate_energy_transfer(self, time_points, initial_state=None):
        """
        Simulate energy transfer dynamics in the coupled system.
        """
        hbar_eV_fs = 0.6582  # hbar in eV*fs
        
        # Initialize
        if initial_state is None:
            initial_state = np.zeros(self.n_total, dtype=complex)
            initial_state[0] = 1.0  # Excitation on OPV site 0, PSU site 0 (tensor product)
        
        states = [initial_state.astype(complex)]
        current_state = initial_state.astype(complex)
        
        # Time evolution
        for i in range(1, len(time_points)):
            dt = time_points[i] - time_points[i-1]
            # Time evolution operator: U(t) = exp(-iHt/hbar)
            U_t = expm(-1j * self.H_total * dt / hbar_eV_fs)
            evolved_state = U_t @ current_state
            states.append(evolved_state.copy())
            current_state = evolved_state
        
        # Calculate population dynamics
        opv_populations = np.zeros((len(states), self.n_opv_sites))
        psu_populations = np.zeros((len(states), self.n_psu_sites))
        
        for i, state in enumerate(states):
            state_matrix = state.reshape((self.n_opv_sites, self.n_psu_sites))
            
            # Trace over PSU to get OPV populations
            for opv_idx in range(self.n_opv_sites):
                opv_pop = 0.0
                for psu_idx in range(self.n_psu_sites):
                    opv_pop += np.abs(state_matrix[opv_idx, psu_idx])**2
                opv_populations[i, opv_idx] = opv_pop
            
            # Trace over OPV to get PSU populations
            for psu_idx in range(self.n_psu_sites):
                psu_pop = 0.0
                for opv_idx in range(self.n_opv_sites):
                    psu_pop += np.abs(state_matrix[opv_idx, psu_idx])**2
                psu_populations[i, psu_idx] = psu_pop
        
        return states, opv_populations, psu_populations