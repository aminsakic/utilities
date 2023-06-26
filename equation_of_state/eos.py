"""
Convenient functions for fitting the energy-volume calculations.
"""

from scipy.optimize import curve_fit
import numpy as np


def murnaghan_fit(volumes: np.ndarray, energies: np.ndarray):
    """ This function performs a Murnaghan EoS fit, and returns 
    the equilibrium properties.

    Parameters
    ----------
    volumes : np.ndarray
        A sequence of volumes.
    energies : np.ndarray
        A sequence of energies, corresponding to each volume.

    Returns
    -------
    Tuple
        E_eq, V_eq, B_eq, Bp_eq, V_fit, E_fit 

        V_fit and E_fit are the volumes generated for the fit and 
        the energies calculated with them. They can be used to plot the fitted curve. 
    """

    # Defining the Murnaghan equation of state function
    def murnaghan_eos(V, E0, V0, B0, Bp):
        return E0 + B0*V0*(1/(Bp*(Bp-1))*(V/V0)**(1-Bp) + V/(Bp*V0) - 1/(Bp-1))

    # Finding initial guesses for V0 and E0
    E0 = np.min(energies)
    min_index = np.argmin(energies)
    V0 = volumes[min_index]

    # Initial guesses
    p_in = [E0, V0, 1, 3.5]

    # Fitting the data to the Murnaghan equation of state
    p_opt, p_cov = curve_fit(murnaghan_eos, volumes, energies, p0=p_in)

    # Extracting fitted parameters
    E_eq = p_opt[0]
    V_eq = p_opt[1]
    B_eq = p_opt[2]
    Bp_eq = p_opt[3]

    # Calculating the standard deviation of the fitted parameters
    p_err = np.sqrt(np.diag(p_cov))

    print(f'Standard deviation of E0, V0, B0, Bp = {p_err}')

    # Preparing the values for the plot
    V_fit = np.linspace(np.min(volumes), np.max(volumes), 100)
    E_fit = murnaghan_eos(V_fit, E_eq, V_eq, B_eq, Bp_eq)

    return E_eq, V_eq, B_eq, Bp_eq, V_fit, E_fit

###############################################################################


def birch_murnaghan_fit(volumes: np.ndarray, energies: np.ndarray):
    """ This function performs a Birch-Murnaghan EoS fit, and returns 
    the equilibrium properties.

    Parameters
    ----------
    volumes : np.ndarray
        A sequence of volumes.
    energies : np.ndarray
        A sequence of energies, corresponding to each volume.

    Returns
    -------
    Tuple
        E_eq, V_eq, B_eq, Bp_eq, V_fit, E_fit 

        V_fit and E_fit are the volumes generated for the fit and 
        the energies calculated with them. They can be used to plot the fitted curve. 
    """

    # Defining the Birch-Murnaghan equation of state function
    def birch_murnaghan_eos(V, E0, V0, B0, Bp):
        return E0 + (9*B0*V0/16)*(((V0/V)**(2/3)-1)**3*Bp + ((V0/V)**(2/3)-1)**2*(6-4*(V0/V)**(2/3)))

    # Finding V0 and E0
    E0 = np.min(energies)
    min_index = np.argmin(energies)
    V0 = volumes[min_index]

    # Initial guesses
    p_in = [E0, V0, 1, 3.5]

    # Fitting the data to the Murnaghan equation of state
    p_opt, p_cov = curve_fit(birch_murnaghan_eos, volumes, energies, p0=p_in)

    # Extracting fitted parameters
    E_eq = p_opt[0]
    V_eq = p_opt[1]
    B_eq = p_opt[2]
    Bp_eq = p_opt[3]

    # Calculating the standard deviation of the fitted parameters
    p_err = np.sqrt(np.diag(p_cov))

    print(f'Standard deviation of E0, V0, B0, Bp = {p_err}')

    # Preparing the values for the plot
    V_fit = np.linspace(np.min(volumes), np.max(volumes), 100)
    E_fit = birch_murnaghan_eos(V_fit, E_eq, V_eq, B_eq, Bp_eq)

    return E_eq, V_eq, B_eq, Bp_eq, V_fit, E_fit

#################################################################################


def polynomial_fit(volumes: np.ndarray, energies: np.ndarray, order=3):
    """ This function performs a polynomial fit, and returns 
    the equilibrium properties.

    Parameters
    ----------
    volumes : np.ndarray
        A sequence of volumes.
    energies : np.ndarray
        A sequence of energies, corresponding to each volume.
    order : int
        Order of the polynomial. Should be higher than 2, because
        the E-V curves typically don't have a quadratic shape! 

    Returns
    -------
    Tuple
        E_eq, V_eq, B_eq, V_fit, E_fit 

        V_fit and E_fit are the volumes generated for the fit and 
        the energies calculated with them. They can be used to plot the fitted curve. 
    """

    # Extract the smallest and the biggest volume
    V_min = np.amin(volumes)
    V_max = np.amax(volumes)

    # Polynomial fit of the data
    coefficients = np.polyfit(volumes, energies, order)

    # Calculate the coefficients of the first derivative
    first_derivative = np.polyder(coefficients, 1)

    # Calculate at which volume the first derivative is zero
    roots = np.roots(first_derivative)

    # Now we need to extract indices of the correct
    # minimum volume from roots
    v_min_indices = np.where((V_min <= roots) & (roots <= V_max))

    # With v_min_indices we can extract the correct minimum
    # volume from roots
    V_eq = float(roots[v_min_indices])

    # Further, we can now evaluate the polynomial at V_eq
    # to get the equilubrium energy
    E_eq = np.polyval(coefficients, V_eq)

    # Now, we will calculate the bulk modulus, B0
    # In general, B = - V*(dp/dV)
    # p(V) = first derivative, hence dp/dV is the second
    # derivative of the energy polynomial
    second_derivative = np.polyder(coefficients, 2)

    # Calculating the bulk modulus
    B_eq = -V_eq * (-np.polyval(second_derivative, V_eq))

    # Preparing the values for the plot
    V_fit = np.linspace(np.min(volumes), np.max(volumes), 100)
    E_fit = np.polyval(coefficients, V_fit)

    return E_eq, V_eq, B_eq, V_fit, E_fit
