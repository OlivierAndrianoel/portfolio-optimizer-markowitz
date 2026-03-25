# Portfolio Optimizer (Markowitz Model)

## Overview

As a Market Finance student, I developed this algorithmic portfolio allocation tool to bridge Modern Portfolio Theory (MPT) with practical Python implementation. 

This engine is designed to ingest historical market data for a basket of equities, compute the covariance matrix, and solve the convex optimization problem to find the Efficient Frontier. It identifies the Optimal Risky Portfolio (Max Sharpe Ratio) and compares its historical performance against a chosen benchmark.

## Key Features

* **Algorithmic Asset Allocation:** Uses Sequential Least Squares Programming (SLSQP) via `SciPy` to determine optimal asset weights, strictly bounded between 0 and 1 (long-only portfolio) with a sum constraint of 100%.
* **Advanced Risk & Return Metrics:** Automatically calculates key quantitative indicators including Annualized Volatility, Sharpe Ratio, Beta, Jensen's Alpha, and the Treynor Ratio.
* **Automated Reporting (Tear Sheet):** Integrates the `QuantStats` library to generate a comprehensive, industry-standard HTML performance report (`tearsheet_group_portfolio.html`), mirroring the analytics used by professional hedge funds.
* **Data Pipeline:** Robust historical data downloading and preprocessing (handling of daily returns and cumulative compounding) using `yfinance` and `Pandas`.

## Technical Stack

* **Language:** Python 3.10+
* **Core Libraries:** * `SciPy` (Mathematical optimization)
  * `NumPy` & `Pandas` (Matrix operations and time-series analysis)
  * `QuantStats` (Financial reporting and metrics)
  * `Matplotlib` (Custom dashboard plotting)

## How to Run the Project

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/OlivierAndrianoel/portfolio-optimizer-markowitz.git](https://github.com/OlivierAndrianoel/portfolio-optimizer-markowitz.git)