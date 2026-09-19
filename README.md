# Portfolio Optimiser Tool

An interactive Python application for exploring portfolio risk, optimising asset allocations, and testing their performance on unseen historical data. Built with Streamlit.

# Features

- Select preset assets or enter custom ticker symbols.
- Download historical adjusted prices using yfinance.
- Choose the historical period, training/test split, and annual risk-free rate.
- Simulate portfolios with different asset weights using Monte Carlo simulation.
- Find maximum-Sharpe and minimum-volatility portfolios.
- Compare buy-and-hold performance against an equal-weight portfolio.
- Display portfolio allocations, risk-return charts, test-period returns, and maximum drawdown.

# How it works

# Screenshots

### Asset selection and settings
![Asset selection and settings](images/Analysis%20settings.png)

### Analysis settings and latest prices
![Analysis settings and latest closing prices](images/Analysis%20settings%20and%20latest%20closing%20prices.png)

### Training statistics and portfolio weights
![Training period stats and optimised portfolios](images/Training%20period%20stats%20and%20optimised%20portfolios.png)

### Monte Carlo simulation
![Monte Carlo results](images/Monte%20carlo%20reults.png)

### Backtest performance
![Performance during test and backtest results](images/Performance%20during%20test%20and%20backtest%20reults.png)

The app splits historical returns chronologically into training and test periods.

Training data is used to estimate annualised returns and covariance, simulate portfolios, and optimise allocations. Portfolio weights must total 100%, with no short selling.

The resulting allocations are then evaluated over the later test period using a buy-and-hold approach. Initial allocations are held without rebalancing, so portfolio weights drift as asset prices change.

An equal-weight portfolio provides a comparison using the same assets and test period.

# Run locally

Install the required packages:

```bash
python -m pip install streamlit yfinance numpy scipy matplotlib pandas
```

From the project folder, launch the app:

```bash
python -m streamlit run app.py
```

An internet connection is required to download price data.

# Project files

- `app.py` — Streamlit interface, data loading, optimisation, charts, and backtesting.
- `portfolio.py` — reusable functions for asset statistics, portfolio volatility, Sharpe ratio, simulations, and drawdown.

# Assumptions and limitations

- Annualisation assumes 252 trading days per year.
- Assets should be quoted in the same currency; currency conversion is not implemented.
- The backtest excludes transaction costs, taxes, and slippage.
- Historical estimates may not reflect future market conditions.
- Optimised portfolios may underperform equal weighting on unseen data.
- Results depend on the selected assets, date range, and available Yahoo Finance data.

This is an educational project, not investment advice.