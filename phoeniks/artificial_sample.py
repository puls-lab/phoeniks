import numpy as np
import sympy as sp
from scipy.constants import c
from collections.abc import Mapping


def gaussian(time, sigma, mu):
    """Simple gaussian function."""
    g = 1 / (sigma + np.sqrt(2 * np.pi))
    g *= np.exp(- 0.5 * (time - mu) ** 2 / sigma ** 2)
    return g


def transfer_function(x, omega, dielectric_model, thickness=None):
    """Based on the transfer function introduced by:
     fit@TDS software from the THzbiophotonics group from Romain Peretti
    https://github.com/THzbiophotonics/Fit-TDS

    This software is based on the publication:
    > "THz-TDS time-trace analysis for the extraction of material and metamaterial parameters"
    from
    > Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet,
      Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and and Nabil Vindas
    at
    > IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2
    > DOI: 10.1109/TTHZ.2018.2889227
    """
    if thickness is None:
        thickness = x.pop(0)
    complex_n = np.sqrt(dielectric_model.evaluate_eps_np(omega, x))

    # Calculation of all the transmission and reflection coefficients
    t12 = 2 / (1 + complex_n)
    t21 = 2 * complex_n / (1 + complex_n)
    r22 = (complex_n - 1) / (1 + complex_n)
    r22b = r22

    # In case we have just two interfaces
    rr = r22 * r22b
    tt = t12 * t21

    propagation = np.exp(-1j * omega * complex_n * thickness / c)
    propagation_air = np.exp(-1j * omega * thickness / c)
    FP = 1 / (1 - rr * (propagation ** 2))
    Z = tt * propagation * FP / propagation_air
    return Z


class Dielectric:
    """A class to save the symbolic expression of the dielectric function of a material.
    A numerical equivalent for the optimization is also provided and stored
    (only gets converted when the dielectric function is updated)."""

    def __init__(self, eps_inf):
        """Create an instance of the Dielectric-class.

        Input:
        eps_inf (float): Provide a constant value of the dielectric function.

        Further class-variables:
        self.variables (dict) : Keeps track of all variable names used by sympy and their current guess value.
        self.components (dict) : Keeps track of how many oscillators (of different types) are put into the dielectric function.
        self.omega (sp.symbol): The dependent variable of the dielectric function [circular frequency]
        self.eps_function (sp.expr): SymPy's symbolic expression of the dielectric function
        self.symbols (sp.symbols): A list of SymPy variables/symbols used in the dielectric function (Later necessary for substitution).
        self.general_np (numpy func): A numpy function based on the symbolic expression, it is dependent on omega and all symbols from sp.symbols.
        """
        self.variables = {"\epsilon_{\infty}": eps_inf}
        self.components = {"base": 1,
                           "lorentz": 0,
                           "debye": 0}
        self.omega = sp.symbols("\omega")
        epsilon_inf = sp.symbols("\epsilon_{\infty}")
        self.eps_function = epsilon_inf
        self.symbols = [self.omega, epsilon_inf]
        self.general_np = sp.lambdify(self.eps_function, self.eps_function)

    def add_lorentz(self, central_frequency, damping_rate, strength):
        """Adds a lorentz oscillator to the dielectric function.

        central_frequency (np.array[float]): Central frequency of the lorentz oscillator in [Hz].
        damping_rate (np.array[float]): Damping rate of the lorentz oscillator in [Hz].
        strength (np.array[float]): Strength of the lorentz oscillator."""

        # Make sure that also single numbers are in a numpy array
        central_frequency = np.array([central_frequency]).ravel()
        damping_rate = np.array([damping_rate]).ravel()
        strength = np.array([strength]).ravel()

        # Count the amount of oscillators and see, if already oscillators were added beforehand
        number_of_oscillator = len(central_frequency)
        start = self.components["lorentz"] + 1
        self.components["lorentz"] += number_of_oscillator

        # Create symbolic variables
        omega_0 = sp.symbols(f"\omega_{{{start}:{start + number_of_oscillator}}}", seq=True)
        gamma = sp.symbols(f"\gamma_{{{start}:{start + number_of_oscillator}}}", seq=True)
        chi = sp.symbols(f"\chi_{{{start}:{start + number_of_oscillator}}}", seq=True)
        self.symbols = [*self.symbols, *omega_0, *gamma, *chi]
        # Add each oscillator to eps_function
        for i in range(number_of_oscillator):
            self.eps_function += chi[i] * omega_0[i] ** 2 / (
                    omega_0[i] ** 2 - self.omega ** 2 + sp.I * self.omega * gamma[i])
            k = start + i
            self.variables.update({f"\omega_{{{k}}}": central_frequency[i]})
            self.variables.update({f"\gamma_{{{k}}}": damping_rate[i]})
            self.variables.update({f"\chi_{{{k}}}": strength[i]})
        self.general_np = sp.lambdify(self.symbols, self.eps_function)

    def add_debye(self, plasma_frequency, damping_rate):
        """Adds a debye oscillator to the dielectric function.

        plasma_frequency (np.array[float]): Plasma frequency of the debye oscillator in [Hz].
        damping_rate (np.array[float]): Damping rate of the debye oscillator in [Hz]."""
        plasma_frequency = np.array([plasma_frequency]).ravel()
        damping_rate = np.array([damping_rate]).ravel()

        number_of_oscillator = len(plasma_frequency)
        start = self.components["debye"] + 1
        self.components["debye"] += number_of_oscillator

        # Create symbolic variables
        omega_p = sp.symbols(f"\omega_{{p{start}:{start + number_of_oscillator}}}", seq=True)
        gamma_p = sp.symbols(f"\gamma_{{p{start}:{start + number_of_oscillator}}}", seq=True)
        self.symbols = [*self.symbols, *omega_p, *gamma_p]
        # Add each oscillator to eps_function
        for i in range(number_of_oscillator):
            self.eps_function += omega_p[i] ** 2 / (self.omega ** 2 + sp.I * self.omega * gamma_p[i])
            k = start + i
            self.variables.update({f"\omega_{{p{k}}}": plasma_frequency[i]})
            self.variables.update({f"\gamma_{{p{k}}}": damping_rate[i]})
        self.general_np = sp.lambdify(self.symbols, self.eps_function)

    def evaluate_eps_np(self, omega, x):
        """Evaluates the numerical expression of the dielectric function.

        Input:
        omega (np.array[float]): Dependent variable, circular frequency.
        x (np.array[float]): List of numerical values, fitting to the list of symbols.
        """
        input_values = [x[str(y).replace("\\\\", "\\")] for y in self.symbols[1:]]
        return self.general_np(omega, *input_values)

    def get_eps_function(self, omega=None):
        if omega is not None:
            eps_np = sp.lambdify(self.omega, self.eps_function.subs(self.variables))
            result = eps_np(omega)
            return result
        else:
            return self.eps_function

    def get_eps_guess(self):
        return self.variables

    def get_n(self, omega=None):
        eps = self.get_eps_function(omega)
        if omega is None:
            n = sp.sqrt((abs(eps) + sp.re(eps)) / 2)
        else:
            n = np.sqrt((np.abs(eps) + eps.real) / 2)
        return n

    def get_kappa(self, omega=None):
        eps = self.get_eps_function(omega)
        if omega is None:
            # We need to use sympy's function if we want to keep the symbolic expression
            k = sp.sqrt((abs(eps) - sp.re(eps)) / 2)
        else:
            k = np.sqrt((np.abs(eps) - eps.real) / 2)
        return k

    def get_alpha(self, omega=None):
        """Get absorption coefficient alpha, already scaled to  [cm^-1]"""
        k = self.get_kappa(omega)
        if omega is None:
            omega = self.omega
        alpha = 2 * omega * k / c
        return alpha


class Dielectric_Model_np:
    """Class to quickly, numerically evaluate a dielectric model.

    Only works with lorentz-based dielectric functions at the moment."""

    def __init__(self):
        pass

    def evaluate_eps_np(self, omega, x):
        if isinstance(x, Mapping):
            x = list(x.values())
        else:
            x = list(x)
        eps = x.pop(0) * np.ones(len(omega)) + 1j * 0
        if len(x) > 0:
            x = np.reshape(np.array(x), (-1, 3))
            omega_0 = x[:, 0]
            gamma = x[:, 1]
            chi = x[:, 2]
            for i in range(x.shape[0]):
                eps += chi[i] * omega_0[i] ** 2 / (omega_0[i] ** 2 - omega ** 2 + 1j * omega * gamma[i])
        return eps
