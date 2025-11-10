import numpy as np

def generate_volatility(V0, xi, kappa, theta, N, dt):
    V = np.empty(N)
    V[0] = V0
    for i in range(1, N):
        dw_v = np.sqrt(dt) * np.random.normal()
        sigma = max(0, V[i-1] + kappa * (theta - V[i-1]) * dt + xi * np.sqrt(V[i-1]) * dw_v)
        V[i] = sigma
    return V

def generate_prices(vola_values, S0, mu, dt, N):
    S = np.empty(N)
    S[0] = S0
    for i in range(1, N):
        dw_s = np.sqrt(dt) * np.random.normal()
        s = S[i-1] * np.exp((mu - 0.5 * vola_values[i]) * dt + np.sqrt(vola_values[i]) * dw_s)
        S[i] = s
    return S