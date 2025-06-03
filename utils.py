import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class CopelandGalaiCalc:
    """
    
    """
    def __init__(self):
        pass


    def bid_ask_normal_distribuition(self, mu:float=100, sigma:float=10, pi:float=0.3) -> tuple:
        """
        This methods ...
        parameters:
        return:
        """
        # Función de beneficio esperado (que queremos igualar a cero)
        def expected_profit(S):
            a = mu + S / 2
            b = mu - S / 2

            z_a = (a - mu) / sigma
            z_b = (mu - b) / sigma

            E_V_gt_a = mu + sigma * norm.pdf(z_a) / (1 - norm.cdf(z_a))
            E_V_lt_b = mu - sigma * norm.pdf(z_b) / norm.cdf(z_b)

            profit_ask = (1 - pi)*(a - mu) + pi*(a - E_V_gt_a)
            profit_bid = (1 - pi)*(mu - b) + pi*(E_V_lt_b - b)

            expected_profit = 0.5 * (profit_ask + profit_bid)
            return expected_profit

        # Resolver el spread de equilibrio (donde beneficio esperado = 0)
        initial_guess = 1.0
        spread_equilibrium = fsolve(expected_profit, initial_guess)[0]

        # Calcular precios de bid y ask
        ask = mu + spread_equilibrium / 2
        bid = mu - spread_equilibrium / 2

        print(f"Spread de equilibrio: {spread_equilibrium:.4f}")
        print(f"Precio Ask: {ask:.4f}")
        print(f"Precio Bid: {bid:.4f}")

        return (bid, ask)
    

    def bid_ask_exponential_distribuition(self, lam:float=0.5, pi:float=0.3):
        
        E_V = 1 / lam

        def expected_profit(S):
            a = E_V + S / 2
            b = E_V - S / 2

            E_V_gt_a = a + (1 / lam)
            E_V_lt_b = (1 / lam) - (b * np.exp(-lam * b)) / (1 - np.exp(-lam * b))

            profit_ask = (1 - pi)*(a - E_V) + pi*(a - E_V_gt_a)
            profit_bid = (1 - pi)*(E_V - b) + pi*(E_V_lt_b - b)

            expected_profit = 0.5 * (profit_ask + profit_bid)
            return expected_profit

        # Resolver el spread de equilibrio (donde beneficio esperado = 0)
        initial_guess = 1.0
        spread_equilibrium = fsolve(expected_profit, initial_guess)[0]

        # Calcular precios de bid y ask
        ask = E_V + spread_equilibrium / 2
        bid = E_V - spread_equilibrium / 2

        print(f"Spread de equilibrio: {spread_equilibrium:.4f}")
        print(f"Precio Ask: {ask:.4f}")
        print(f"Precio Bid: {bid:.4f}")

        return (bid, ask)