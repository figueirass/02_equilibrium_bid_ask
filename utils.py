import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class CopelandGalaiCalc:
    """
    Cálculo del spread de equilibrio y precios bid/ask bajo el modelo
    de Copeland & Galai (1983).

    El market maker fija un spread S tal que su beneficio esperado es cero
    frente a la posibilidad de operar con traders informados (probabilidad π).

    Soporta distribución Normal (μ, σ²) y Exponencial (λ).
    """

    def __init__(self):
        pass


    # ---------- Distribución Normal ----------
    def bid_ask_normal_distribuition(self, mu:float=100, sigma:float=10, pi:float=0.3) -> tuple:
        """
        Equilibrium bid/ask for a normally-distributed fundamental value.

        Parameters
        ----------
        mu : float
            Mean of the fundamental value V.
        sigma : float
            Standard deviation of V.
        pi : float
            Probability that the counterparty is informed.
        initial_guess : float
            Initial guess for the non-linear solver.

        Returns
        -------
        bid : float
        ask : float
        spread : float  (ask - bid)
        """

        def expected_profit(S):
            a = mu + S / 2  # ask quote
            b = mu - S / 2  # bid quote

            z_a = (a - mu) / sigma
            z_b = (mu - b) / sigma

            # Conditional expectations E[V | V > a] and E[V | V < b]
            E_V_gt_a = mu + sigma * norm.pdf(z_a) / (1 - norm.cdf(z_a))
            E_V_lt_b = mu - sigma * norm.pdf(z_b) / norm.cdf(z_b)

            profit_ask = (1 - pi)*(a - mu) + pi*(a - E_V_gt_a)
            profit_bid = (1 - pi)*(mu - b) + pi*(E_V_lt_b - b)

            expected_profit = 0.5 * (profit_ask + profit_bid)   # MM trades ask & bid equally
            return expected_profit

        initial_guess = 1.0
        spread_equilibrium = fsolve(expected_profit, initial_guess)[0]

        ask = mu + spread_equilibrium / 2
        bid = mu - spread_equilibrium / 2

        print(f"Spread de equilibrio: {spread_equilibrium:.4f}")
        print(f"Precio Ask: {ask:.4f}")
        print(f"Precio Bid: {bid:.4f}")

        return (bid, ask, spread_equilibrium)
    

    # ---------- Distribución Exponencial ----------
    def bid_ask_exponential_distribuition(self, lam:float=0.5, pi:float=0.3):
        """
        Equilibrium bid/ask for an exponentially-distributed fundamental value.

        Parameters
        ----------
        lam : float
            Rate parameter λ (mean = 1/λ).
        pi : float
            Probability that the counterparty is informed.
        initial_guess : float
            Initial guess for the non-linear solver.

        Returns
        -------
        bid : float
        ask : float
        spread : float  (ask - bid)
        """

        E_V = 1 / lam

        def expected_profit(S):
            a = E_V + S / 2
            b = E_V - S / 2

            # Conditional means for the exponential distribution
            E_V_gt_a = a + (1 / lam)
            E_V_lt_b = (1 / lam) - (b * np.exp(-lam * b)) / (1 - np.exp(-lam * b))

            profit_ask = (1 - pi)*(a - E_V) + pi*(a - E_V_gt_a)
            profit_bid = (1 - pi)*(E_V - b) + pi*(E_V_lt_b - b)

            expected_profit = 0.5 * (profit_ask + profit_bid)
            return expected_profit

        initial_guess = 1.0
        spread_equilibrium = fsolve(expected_profit, initial_guess)[0]

        ask = E_V + spread_equilibrium / 2
        bid = E_V - spread_equilibrium / 2

        print(f"Spread de equilibrio: {spread_equilibrium:.4f}")
        print(f"Precio Ask: {ask:.4f}")
        print(f"Precio Bid: {bid:.4f}")

        return (bid, ask, spread_equilibrium)