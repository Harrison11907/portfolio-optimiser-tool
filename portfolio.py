import numpy as np


# --------------------
# ASSET STATISTICS
# --------------------

def calculate_asset_statistics(returns_data, trading_days):
    annual_returns = returns_data.mean() * trading_days
    annual_volatility = returns_data.std(ddof=1) * np.sqrt(trading_days)
    annual_covariance = returns_data.cov() * trading_days

    return annual_returns, annual_volatility, annual_covariance

# --------------------
# OPTIMISATION FUNCTIONS
# --------------------

def portfolio_volatility_func(weights, annual_covariance):
    portfolio_variance = weights @ annual_covariance @ weights
    return np.sqrt(portfolio_variance)


def negative_sharpe(weights, annual_returns, annual_covariance, risk_free_rate):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = portfolio_volatility_func(weights, annual_covariance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return -sharpe_ratio

# --------------------
# MAXIMUM DRAWDOWN
# --------------------

def maximum_drawdown(growth):
    running_peak = growth.cummax().clip(lower=1)
    drawdown = growth / running_peak - 1
    return drawdown.min()

# --------------------
# MONTE CARLO PORTFOLIOS
# --------------------

def simulate_portfolios(annual_returns, annual_covariance, num_portfolios):
    num_assets = len(annual_returns)

    rng = np.random.default_rng(42)

    weights = rng.random((num_portfolios, num_assets))
    weights = weights / weights.sum(axis=1, keepdims=True)

    mean_returns = annual_returns.to_numpy()
    covariance = annual_covariance.to_numpy()

    portfolio_returns = weights @ mean_returns

    portfolio_variances = np.sum(
        (weights @ covariance) * weights,
        axis=1)

    portfolio_volatilities = np.sqrt(
        np.maximum(portfolio_variances, 0))

    return portfolio_returns, portfolio_volatilities