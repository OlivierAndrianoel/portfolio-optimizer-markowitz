import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import quantstats as qs
import warnings
import logging
from typing import Dict, Tuple
from scipy.stats import norm

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    def __init__(self, tickers: list, benchmark: str, start_date: str, end_date: str, rf_rate: float,
                 market_return: float):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.rf_rate = rf_rate
        self.market_return = market_return

        self.data = pd.DataFrame()
        self.bench_data = pd.Series()
        self.weights = {}

    def fetch_and_prepare_data(self) -> None:
        logger.info(f"Downloading daily data for {len(self.tickers)} assets and the benchmark...")

        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        bench_data = yf.download(self.benchmark, start=self.start_date, end=self.end_date)['Close']

        self.data = raw_data.ffill().dropna()

        if isinstance(bench_data, pd.DataFrame):
            self.bench_data = bench_data.squeeze()
        else:
            self.bench_data = bench_data.ffill().dropna()

        self.data, self.bench_data = self.data.align(self.bench_data, join='inner', axis=0)

    def optimize_portfolio(self) -> Dict[str, float]:
        logger.info("Calculating Markowitz optimization (Annualized daily data)...")

        returns = self.data.pct_change().dropna()

        self.mu = returns.mean() * 252
        self.S = returns.cov() * 252

        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(0.0, 0.20))
        self.ef_plot = EfficientFrontier(self.mu, self.S, weight_bounds=(0.0, 0.20))

        raw_weights = ef.max_sharpe(risk_free_rate=self.rf_rate)
        self.weights = ef.clean_weights()

        logger.info("Optimal weights found.")
        return self.weights

    def calculate_returns(self) -> Tuple[pd.Series, pd.Series]:
        logger.info("Calculating portfolio performance...")

        returns = self.data.pct_change().dropna()
        bench_returns = self.bench_data.pct_change().dropna()

        portfolio_returns = returns.dot(pd.Series(self.weights))
        portfolio_returns.name = "Group_Portfolio"

        portfolio_returns, bench_returns = portfolio_returns.align(bench_returns, join='inner')
        return portfolio_returns, bench_returns

    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        return returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_sharpe(returns: pd.Series, rf_rate: float) -> float:
        annualized_return = returns.mean() * 252
        annualized_vol = PortfolioOptimizer.calculate_volatility(returns)
        return (annualized_return - rf_rate) / annualized_vol

    @staticmethod
    def calculate_beta(port_returns: pd.Series, bench_returns: pd.Series) -> float:
        cov = port_returns.cov(bench_returns)
        var = bench_returns.var()
        return cov / var

    @staticmethod
    def calculate_treynor(returns: pd.Series, beta: float, rf_rate: float) -> float:
        annualized_return = returns.mean() * 252
        return (annualized_return - rf_rate) / beta

    @staticmethod
    def calculate_alpha(returns: pd.Series, beta: float, rf_rate: float, market_return: float) -> float:
        annualized_return = returns.mean() * 252
        return annualized_return - (rf_rate + beta * (market_return - rf_rate))

    @staticmethod
    def calculate_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
        mu = np.mean(returns)
        sigma = np.std(returns)
        return norm.ppf(1 - confidence, mu, sigma)

    @staticmethod
    def calculate_var_monte_carlo(returns: pd.Series, confidence: float = 0.95, simulations: int = 10000) -> float:
        mu = np.mean(returns)
        sigma = np.std(returns)
        np.random.seed(42)
        simulated_returns = np.random.normal(mu, sigma, simulations)
        return np.percentile(simulated_returns, (1 - confidence) * 100)

if __name__ == "__main__":

    TICKERS = ['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'MA', 'META', 'MSFT', 'NVDA', 'ORCL', 'TSLA']
    BENCHMARK = 'SEEGX'
    START_DATE = "2020-09-01"
    END_DATE = "2025-10-01"
    RF_RATE = 0.04
    MARKET_RETURN = 0.15

    optimizer = PortfolioOptimizer(TICKERS, BENCHMARK, START_DATE, END_DATE, RF_RATE, MARKET_RETURN)
    optimizer.fetch_and_prepare_data()
    weights = optimizer.optimize_portfolio()

    port_returns, bench_returns = optimizer.calculate_returns()

    port_vol = optimizer.calculate_volatility(port_returns)
    port_sharpe = optimizer.calculate_sharpe(port_returns, RF_RATE)
    beta = optimizer.calculate_beta(port_returns, bench_returns)
    alpha = optimizer.calculate_alpha(port_returns, beta, RF_RATE, MARKET_RETURN)
    treynor = optimizer.calculate_treynor(port_returns, beta, RF_RATE)
    port_var_hist = optimizer.calculate_var_historical(port_returns)
    port_var_param = optimizer.calculate_var_parametric(port_returns)
    port_var_mc = optimizer.calculate_var_monte_carlo(port_returns)

    bench_vol = optimizer.calculate_volatility(bench_returns)
    bench_sharpe = optimizer.calculate_sharpe(bench_returns, RF_RATE)
    bench_var_hist = optimizer.calculate_var_historical(bench_returns)

    logger.info("Generating comprehensive dashboard plot...")

    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Portfolio Optimization Dashboard", fontsize=16, fontweight='bold', y=0.95)

    plotting.plot_efficient_frontier(optimizer.ef_plot, ax=ax[0, 0], show_assets=True)
    ef_for_perf = EfficientFrontier(optimizer.mu, optimizer.S, weight_bounds=(0.0, 0.20))
    ef_for_perf.set_weights(weights)
    ret_tangent, std_tangent, _ = ef_for_perf.portfolio_performance(risk_free_rate=RF_RATE)

    ax[0, 0].scatter(std_tangent, ret_tangent, marker="*", s=200, c="red", zorder=10, label="Max Sharpe")
    ax[0, 0].set_title("Efficient Frontier")
    ax[0, 0].legend()

    # --- Plot 2: Allocation Pie Chart (Top Right) ---
    pd.Series(weights).plot.pie(ax=ax[0, 1], autopct='%1.1f%%', cmap='tab20')
    ax[0, 1].set_title("Group Portfolio Allocation (Max 20% constraint)")
    ax[0, 1].set_ylabel("")

    INITIAL_INVESTMENT = 10000
    cumulative_port = (1 + port_returns).cumprod() * INITIAL_INVESTMENT
    cumulative_bench = (1 + bench_returns).squeeze().cumprod() * INITIAL_INVESTMENT

    cumulative_port.plot(ax=ax[1, 0], label="Group Portfolio", color='blue')
    cumulative_bench.plot(ax=ax[1, 0], label=f"{BENCHMARK} Fund", color='red')
    ax[1, 0].set_title("GROWTH OF $10 000")
    ax[1, 0].set_ylabel("Value ($)")
    ax[1, 0].legend()

    ax[1, 1].axis('off')
    ax[1, 1].set_title("Performance & Risk Metrics Comparison", fontweight='bold')

    table_data = [
        ["Annualized Volatility", f"{port_vol * 100:.2f}%", f"{bench_vol * 100:.2f}%"],
        ["Sharpe Ratio", f"{port_sharpe:.2f}", f"{bench_sharpe:.2f}"],
        ["Beta", f"{beta:.2f}", "1.00"],
        ["Jensen's Alpha", f"{alpha * 100:.2f}%", "-"],
        ["Treynor Ratio", f"{treynor:.2f}", "-"],
        ["Historical VaR (95%)", f"{port_var_hist * 100:.2f}%", f"{bench_var_hist * 100:.2f}%"],
        ["Parametric VaR (95%)", f"{port_var_param * 100:.2f}%", "-"],
        ["Monte Carlo VaR (95%)", f"{port_var_mc * 100:.2f}%", "-"]
    ]
    columns = ["Metric", "Group Portfolio", f"{BENCHMARK} Fund"]

    table = ax[1, 1].table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("portfolio_results.png", dpi=300)
    logger.info("Dashboard saved as portfolio_results.png")
    plt.show()

    logger.info("Generating QuantStats HTML report...")
    try:
        qs.reports.html(port_returns.squeeze(), benchmark=bench_returns.squeeze(),
                        output='tearsheet_group_portfolio.html', title="Portfolio vs SEEGX Fund")
        logger.info("Report successfully generated: tearsheet_group_portfolio.html")
    except Exception as e:
        logger.error(f"QuantStats Error : {e}")