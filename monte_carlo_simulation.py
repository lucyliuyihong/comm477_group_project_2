import numpy as np
import pandas as pd

# Function to simulate one price path
def simulate_price_path(S0, T, r, sigma, N):
    dt = T / N
    prices = [S0]
    for _ in range(N):
        Z = np.random.normal()
        S_t = prices[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_t)
    return prices

# Function to calculate the payoff for one price path
def calculate_payoff(prices, B, S0):
    final_price = prices[-1]
    if final_price >= 1.6 * S0:
        # If performance is above 160%, receive principal + 100% of the performance above 100%
        return S0 + (final_price - S0) * 1.0
    elif final_price >= S0:
        # If performance is between 100% and 160%, receive principal + 60%
        return S0 + S0 * 0.6
    elif final_price >= 0.8 * S0:
        # If performance is between 80% and 100%, receive the principal back
        return S0
    else:
        # If performance is below 80%, receive the underlying performance
        return final_price

# Monte Carlo simulation function
def monte_carlo_simulation(S0, T, r, sigma, B, M, N):
    payoffs = []
    for _ in range(M):
        path = simulate_price_path(S0, T, r, sigma, N)
        payoff = calculate_payoff(path, B, S0)
        payoffs.append(payoff)
    present_value = np.exp(-r * T) * np.mean(payoffs)
    return present_value

# Parameters
# maturity date = 19-Dec-25
# issue date = 20-Dec-19
# current date = 20-Mar-24 for simplicity
S0 = 100.0  # Initial underlying asset price
T = 1.75    # Time to maturity in years (1 year and 9 months)
r = 0.035  # Risk-free interest rate
B = 80.0    # Barrier level
M = 10000   # Number of simulation paths
N = 252     # Number of time steps

# Load underlying asset's historical price data
file_path = './S&P:TSX Composite Low Volatility Index.xls'
historical_data = pd.read_excel(file_path)

# Convert the 'Date' column to datetime
historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

# Make sure dates are in the correct order, sort by date
historical_data.sort_values('Date', inplace=True)

# Change data type to floar
historical_data['Price'] = historical_data['Price'].astype(float)

# Calculate daily returns
historical_data['Daily Returns'] = historical_data['Price'].pct_change(fill_method=None)

# Calculate annualized volatility
daily_volatility = historical_data['Daily Returns'].std(skipna=True)

# Annuliaze the volatility
annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days in a year

# Perform the Monte Carlo simulation
fair_value = monte_carlo_simulation(S0, T, r, annualized_volatility, B, M, N)
fair_value
