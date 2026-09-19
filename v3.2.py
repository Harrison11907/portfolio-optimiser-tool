import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# --------------------
# SETTINGS - import libraries - and then select assets
# --------------------

def parse_tickers(text):
    tickers = []

    for ticker in text.replace(",", " ").upper().split():
        if ticker not in tickers:
            tickers.append(ticker)

    if not tickers:
        raise ValueError("Enter at least one ticker.")

    return tickers


available_assets = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "SPY": "SPDR S&P 500 ETF",
    "GLD": "SPDR Gold Shares",
}

print("\nExample assets:")
for ticker, name in available_assets.items():
    print(f"  {ticker:<6} — {name}")

print("\nYou can also enter other Yahoo Finance tickers.")

tickers = parse_tickers(input("Enter tickers separated by spaces or commas: "))

num_assets = len(tickers)
print("Selected assets:", tickers)

trading_days = 252


# --------------------
# DOWNLOAD DATA - Apple, Microsoft, Nvidia etc and checking it downloaded
# --------------------

data = yf.download(tickers, period="2y")
closing_prices = data["Close"]

# Match price columns to the user's ticker order.
# Any missing ticker gets an empty column.
closing_prices = closing_prices.reindex(columns=tickers)

# Find assets that have no usable closing prices.
failed_tickers = closing_prices.columns[closing_prices.isna().all()].tolist()

if failed_tickers:
    raise ValueError(
        "No price data returned for: "
        + ", ".join(failed_tickers)
        + ". Check the symbols or try again later.")

# --------------------
# CALCULATE RETURNS - from closing prices
# --------------------

returns = closing_prices.pct_change(fill_method=None).dropna()

if len(returns) < 2:
    raise ValueError(
        "Not enough overlapping return data for the selected assets."
    )


# --------------------
# SPLIT RETURNS INTO TRAINING AND TEST DATA
# --------------------

returns = returns.sort_index()

split_index = int(len(returns) * 0.8)

train_returns = returns.iloc[:split_index]
test_returns = returns.iloc[split_index:]

if len(train_returns) < 2 or len(test_returns) < 2:
    raise ValueError("Not enough data for a training/test split.")

print(
    f"Training: {train_returns.index[0].date()} "
    f"to {train_returns.index[-1].date()}")

print(
    f"Testing:  {test_returns.index[0].date()} "
    f"to {test_returns.index[-1].date()}")


# --------------------
# ASSET STATISTICS - calculated with training data
# --------------------
# remember when calculating std from a sample you lose a degree of freedom so divide by n-1 (ddof=1)

def calculate_asset_statistics(returns_data, trading_days):
    annual_returns = returns_data.mean() * trading_days
    annual_volatility = returns_data.std(ddof=1) * np.sqrt(trading_days)
    annual_covariance = returns_data.cov() * trading_days

    return annual_returns, annual_volatility, annual_covariance


annual_returns, annual_volatility, annual_covariance = calculate_asset_statistics(train_returns, trading_days)

# --------------------
# PORTFOLIO - weights go in order of application to assets eg 0.5 --> Apple
# --------------------

portfolio_returns = []
portfolio_volatilities = []
sharpe_ratios = []
portfolio_weights = []
num_portfolios = 100000
risk_free_rate = 0

for i in range(num_portfolios):

    weights = np.random.random(num_assets)

    weights = weights / np.sum(weights)

    portfolio_return = np.dot(weights, annual_returns)

    portfolio_variance = (weights @ annual_covariance @ weights)

    portfolio_volatility = np.sqrt(portfolio_variance)

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility


    portfolio_returns.append(portfolio_return)
    portfolio_volatilities.append(portfolio_volatility)
    sharpe_ratios.append(sharpe_ratio)
    portfolio_weights.append(weights)


# --------------------
# BEST PERFORMING PORTFOLIOS - find highest sharpe ratio portfolio, lowest volatility portfolio
# --------------------

max_sharpe_index = np.argmax(sharpe_ratios)
max_sharpe_weights = portfolio_weights[max_sharpe_index]
max_sharpe_return = portfolio_returns[max_sharpe_index]
max_sharpe_volatility = portfolio_volatilities[max_sharpe_index]
max_sharpe = sharpe_ratios[max_sharpe_index]

min_volatility_index = np.argmin(portfolio_volatilities)
min_vol_weights = portfolio_weights[min_volatility_index]
min_vol_return = portfolio_returns[min_volatility_index]
min_volatility = portfolio_volatilities[min_volatility_index]
min_vol_sharpe = sharpe_ratios[min_volatility_index]

#---------------------
#SORTING POTFOLIOS
#---------------------

sorted_indices = np.argsort(portfolio_volatilities)
efficient_volatilities = []
efficient_returns = []
best_return_so_far = -np.inf

for i in sorted_indices:
    current_return = portfolio_returns[i]
    current_volatility = portfolio_volatilities[i]

    if current_return > best_return_so_far:
        efficient_volatilities.append(current_volatility)
        efficient_returns.append(current_return)

        best_return_so_far = current_return

#---------------------
#OPTIMISATION - finding the actual minimum possible volatility / max sharpe 
#---------------------

def portfolio_volatility_func(weights, annual_covariance):
    portfolio_variance = weights @ annual_covariance @ weights
    return np.sqrt(portfolio_variance)

initial_guess = initial_guess = np.array([1 / num_assets] * num_assets)

bounds = bounds = tuple((0, 1) for _ in range(num_assets))

constraint = {"type": "eq","fun": lambda weights: np.sum(weights) - 1}

min_vol_result = minimize(portfolio_volatility_func,initial_guess,args=(annual_covariance),bounds=bounds,constraints=constraint)
if not min_vol_result.success:
    raise ValueError(
        "Minimum-volatility optimisation failed: "
        + min_vol_result.message)

def negative_sharpe(weights, annual_returns, annual_covariance, risk_free_rate):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = portfolio_volatility_func(weights, annual_covariance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio


actual_max_sharpe = minimize(negative_sharpe, initial_guess,args=(annual_returns, annual_covariance, risk_free_rate),bounds=bounds,constraints=constraint)
if not actual_max_sharpe.success:
    raise ValueError(
        "Maximum-Sharpe optimisation failed: "
        + actual_max_sharpe.message)

#PORTFOLIO- bringing together the portfolios from the optimiser

opt_min_vol_weights = min_vol_result.x

opt_min_vol_return = np.dot(opt_min_vol_weights,annual_returns)

opt_min_vol_volatility = min_vol_result.fun

opt_max_sharpe_weights = actual_max_sharpe.x

opt_max_sharpe_return = np.dot(opt_max_sharpe_weights,annual_returns)

opt_max_sharpe_volatility = np.sqrt(opt_max_sharpe_weights@ annual_covariance@ opt_max_sharpe_weights)

# --------------------
# BACKTEST - buy and hold during the test period
# --------------------

# Track how 1 unit invested in each asset grows.
# Currency follows the input data; these are growth factors.
asset_growth = (1 + test_returns).cumprod()

# Combine asset growth using the starting portfolio weights.
max_sharpe_growth = asset_growth @ opt_max_sharpe_weights
min_vol_growth = asset_growth @ opt_min_vol_weights

# Compare against investing equally in every selected asset.
equal_weights = np.full(num_assets, 1 / num_assets)
equal_weight_growth = asset_growth @ equal_weights

print("\nTest-period returns:")
print(f"Max Sharpe:   {max_sharpe_growth.iloc[-1] - 1:.2%}")
print(f"Min volatility: {min_vol_growth.iloc[-1] - 1:.2%}")
print(f"Equal weight: {equal_weight_growth.iloc[-1] - 1:.2%}")

#---------------------
# OUTPUT - optimised portfolio weights
#---------------------

print("\nOptimised portfolio weights:")
print(f"{'Asset':<10}{'Max Sharpe':>15}{'Min Volatility':>18}")

for ticker, sharpe_weight, min_vol_weight in zip(tickers, opt_max_sharpe_weights, opt_min_vol_weights):
    print(f"{ticker:<10}{sharpe_weight:>15.2%}{min_vol_weight:>18.2%}")

# --------------------
# BACKTEST STATISTICS - maximum drawdown
# --------------------

def maximum_drawdown(growth):
    running_peak = growth.cummax().clip(lower=1)
    drawdown = growth / running_peak - 1
    return drawdown.min()


print("\nMaximum drawdown:")
print(f"Max Sharpe: {maximum_drawdown(max_sharpe_growth):.2%}")
print(f"Min volatility: {maximum_drawdown(min_vol_growth):.2%}")
print(f"Equal weight: {maximum_drawdown(equal_weight_growth):.2%}")

# --------------------
# OUTPUT - a risk-return scatter plot
# --------------------

plt.scatter(portfolio_volatilities, portfolio_returns, s=2, label ='Simulated Portfolios')
plt.scatter(max_sharpe_volatility, max_sharpe_return, s=100, marker="*", label = 'Max Sharpe')
plt.scatter(min_volatility, min_vol_return, s=100, marker="*", label = 'Min Volatility')
plt.plot(efficient_volatilities, efficient_returns, linewidth=2, color = "red", label="Efficient Frontier")
plt.scatter(opt_min_vol_volatility, opt_min_vol_return, s=100, marker="*", label = 'Optimised Min Volatility')
plt.scatter(opt_max_sharpe_volatility, opt_max_sharpe_return, s=100, marker="*", label= 'Optimised Max Sharpe')



plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Annual Volatility")
plt.ylabel("Expected Annual Return")
plt.title("Monte Carlo Portfolio Simulation")

plt.legend()

# --------------------
# OUTPUT - backtest performance chart
# --------------------

plt.figure()

plt.plot(max_sharpe_growth.index, max_sharpe_growth, label='Max Sharpe')
plt.plot(min_vol_growth.index, min_vol_growth, label='Min Volatility')
plt.plot(equal_weight_growth.index, equal_weight_growth, linestyle='--', label='Equal Weight')

plt.axhline(y=1, color='grey', linestyle=':', linewidth=1)

plt.xlabel("Date")
plt.ylabel("Portfolio Growth")
plt.title("Buy-and-Hold Performance - Test Period")

plt.legend()
plt.show()
