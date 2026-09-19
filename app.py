#--------------------
# IMPORTING RELEVANT LIBRARIES/ MODULES - and functions from portfolio.py
#--------------------

import streamlit as st
import yfinance as yf
from portfolio import (
    calculate_asset_statistics,
    portfolio_volatility_func,
    negative_sharpe,
    maximum_drawdown,
    simulate_portfolios)

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



def parse_tickers(text):
    tickers = []

    for ticker in text.replace(",", " ").upper().split():
        if ticker not in tickers:
            tickers.append(ticker)

    if not tickers:
        raise ValueError("Enter at least one ticker.")

    return tickers

# --------------------
# PAGE TITLE
# --------------------

st.title("Portfolio Optimisation")
st.write("Select assets to analyse and backtest.")

# --------------------
# ASSET SELECTION - automatic asset options
# --------------------

available_assets = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "SPY": "SPDR S&P 500 ETF",
    "GLD": "SPDR Gold Shares",
}

tickers = st.multiselect(
    "Choose your assets",
    options=list(available_assets),
    default=["AAPL", "MSFT", "GLD"]
)

# --------------------
# ADDITIONAL ASSETS - allows other tickers/ assets to be selected
# --------------------

extra_tickers = st.text_input(
    "Other tickers",
    placeholder="For example: AGG, VXUS"
)

if extra_tickers.strip():
    tickers = parse_tickers(
        " ".join(tickers) + " " + extra_tickers
    )

st.write("Selected assets:", tickers)

if not tickers:
    st.warning("Choose at least one asset or enter a ticker.")
    st.stop()

num_assets = len(tickers)

# --------------------
# ANALYSIS SETTINGS - allows user to modify: period of data, training vs testing period, number of portfolios, and risk free rate
# --------------------

period = st.selectbox(
    "Historical data period",
    options=["1y", "2y", "5y", "10y"],
    index=1)

training_percentage = st.slider(
    "Percentage of data used for training",
    min_value=50,
    max_value=90,
    value=80,
    step=5
)

st.write(f"The remaining {100 - training_percentage}% is used for testing.")

# the risk free rate is basically the return you could earn without taking the investment risk we measure
# (risk-free doesn't literally mean zero possible risk, analysts use a very low-risk, govt borrowing rate as a proxy)
risk_free_percentage = st.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.25)

risk_free_rate = risk_free_percentage / 100

num_portfolios = st.slider(
    "Number of simulated portfolios",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000)

# --------------------
# DOWNLOAD DATA - downloading the assets data from yahoo finance and providing errors in st
# --------------------

if not st.button("Run analysis"):
    st.stop()

data = yf.download(tickers,period=period,auto_adjust=True,multi_level_index=True,progress=False)

if data is None or data.empty:
    st.error("No data returned. Check your tickers and connection, then try again.")
    st.stop()

closing_prices = data["Close"].reindex(columns=tickers)

failed_tickers = closing_prices.columns[closing_prices.isna().all()].tolist()

if failed_tickers:
    st.error("No price data returned for: " + ", ".join(failed_tickers))
    st.stop()

st.subheader("Latest adjusted closing prices")
st.dataframe(closing_prices.tail())

# --------------------
# CALCULATE RETURNS - dont fill in missing prices before calculating pct change
# --------------------

returns = closing_prices.pct_change(fill_method=None).dropna()
returns = returns.sort_index()


# --------------------
# SPLIT INTO TRAINING AND TEST DATA - we split the data using some for our, monte carlo and optimised portfolios and we use the rest to test how they perform
# --------------------

split_index = int(len(returns) * training_percentage / 100)

train_returns = returns.iloc[:split_index]
test_returns = returns.iloc[split_index:]

if len(train_returns) < 2 or len(test_returns) < 2:
    st.error("Not enough overlapping data for a training/test split.")
    st.stop()

st.write(f"Training: {train_returns.index[0].date()} "
    f"to {train_returns.index[-1].date()}")

st.write(f"Testing: {test_returns.index[0].date()} "
    f"to {test_returns.index[-1].date()}")


# --------------------
# ASSET STATISTICS - provide stats from the training period
# --------------------

trading_days = 252

annual_returns, annual_volatility, annual_covariance = calculate_asset_statistics(train_returns, trading_days)

statistics = annual_returns.to_frame(name="Annualised mean return")
statistics["Annualised volatility"] = annual_volatility

st.subheader("Training-period asset statistics")
st.dataframe(statistics.style.format("{:.2%}"))

# --------------------
# OPTIMISATION
# --------------------


initial_guess = np.array([1 / num_assets] * num_assets)
bounds = tuple((0, 1) for _ in range(num_assets))
constraint = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

min_vol_result = minimize(
    portfolio_volatility_func,
    initial_guess,
    args=(annual_covariance,),
    bounds=bounds,
    constraints=constraint)

if not min_vol_result.success:
    st.error("Minimum-volatility optimisation failed: " + min_vol_result.message)
    st.stop()

actual_max_sharpe = minimize(
    negative_sharpe,
    initial_guess,
    args=(annual_returns, annual_covariance, risk_free_rate),
    bounds=bounds,
    constraints=constraint)

if not actual_max_sharpe.success:
    st.error("Maximum-Sharpe optimisation failed: " + actual_max_sharpe.message)
    st.stop()

opt_min_vol_weights = min_vol_result.x
opt_max_sharpe_weights = actual_max_sharpe.x


# --------------------
# DISPLAY PORTFOLIO WEIGHTS
# --------------------

weights_table = annual_returns.to_frame(name="Max Sharpe")
weights_table["Max Sharpe"] = opt_max_sharpe_weights
weights_table["Min Volatility"] = opt_min_vol_weights

st.subheader("Optimised portfolio weights")
st.dataframe(weights_table.style.format("{:.2%}"))

# --------------------
# MONTE CARLO SIMULATION - run lots of possible different weights and calculate outcomes eg annual volatility and return
# --------------------


portfolio_returns, portfolio_volatilities = simulate_portfolios(
    annual_returns, annual_covariance, num_portfolios)

opt_max_sharpe_return = np.dot(opt_max_sharpe_weights, annual_returns)
opt_max_sharpe_volatility = portfolio_volatility_func(opt_max_sharpe_weights, annual_covariance)

opt_min_vol_return = np.dot(opt_min_vol_weights, annual_returns)
opt_min_vol_volatility = portfolio_volatility_func(opt_min_vol_weights, annual_covariance)


# --------------------
# OUTPUT - training risk-return chart
# --------------------

st.subheader("Portfolio risk and return — training period")

fig, ax = plt.subplots()

ax.scatter(portfolio_volatilities, portfolio_returns, s=2, alpha=0.3, label='Simulated Portfolios')
ax.scatter(opt_max_sharpe_volatility, opt_max_sharpe_return, s=150, marker='*', color='orange', edgecolors='black', label='Optimised Max Sharpe', zorder=3)
ax.scatter(opt_min_vol_volatility, opt_min_vol_return, s=150, marker='*', color='red', edgecolors='black', label='Optimised Min Volatility', zorder=3)

ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_xlabel("Annual Volatility")
ax.set_ylabel("Annualised Mean Return")
ax.set_title("Monte Carlo Portfolio Simulation")

ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# --------------------
# BACKTEST - buy and hold
# --------------------

asset_growth = (1 + test_returns).cumprod()

max_sharpe_growth = asset_growth @ opt_max_sharpe_weights
min_vol_growth = asset_growth @ opt_min_vol_weights

equal_weights = np.full(num_assets, 1 / num_assets)
equal_weight_growth = asset_growth @ equal_weights


# --------------------
# DISPLAY BACKTEST
# --------------------

growth_table = max_sharpe_growth.to_frame(name="Max Sharpe")
growth_table["Min Volatility"] = min_vol_growth
growth_table["Equal Weight"] = equal_weight_growth

st.subheader("Performance during the test period")
st.line_chart(growth_table)

st.caption(
    "Growth starts from 1: a value of 1.10 means a 10% gain. "
    "The first plotted point includes the first test day's return.")

# --------------------
# BACKTEST RESULTS
# --------------------

backtest_results = (growth_table.iloc[-1] - 1).to_frame(
    name="Test-period return")

backtest_results["Maximum drawdown"] = growth_table.apply(maximum_drawdown)

st.subheader("Backtest results")
st.dataframe(backtest_results.style.format("{:.2%}"))