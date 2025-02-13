import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import json
from tqdm import tqdm

def black_litterman_LLM(data_dict, returns, sp500_tickers, market_equilibrium_return, tau):
    Q = np.array([np.mean(data_dict[ticker]['expected_return']) for ticker in sp500_tickers])
    P = np.eye(len(sp500_tickers))
    Omega = np.diag([np.var(data_dict[ticker]['expected_return']) for ticker in sp500_tickers])

    sigma = np.cov(returns.T)
    tau_sigma = tau * sigma

    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_Omega = np.linalg.pinv(Omega)
    M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_Omega @ P)

    posterior_returns = M @ (inv_tau_sigma @ market_equilibrium_return + P.T @ inv_Omega @ Q)

    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    def objective_function(weights, expected_returns, cov_matrix, risk_aversion=0.1):
        return portfolio_variance(weights, cov_matrix) - risk_aversion * (weights @ expected_returns)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Fully invested portfolio constraint
        {'type': 'ineq', 'fun': lambda x: x}  # No short-selling constraint (weights must be >= 0)
    )

    bounds = tuple((0, 1) for _ in range(len(sp500_tickers)))

    result = minimize(objective_function, np.ones(len(sp500_tickers)) / len(sp500_tickers),
                     args=(posterior_returns, sigma),
                     constraints=constraints, bounds=bounds)

    return result.x

def process_period(start_date, end_date, tau):
    # Load market caps
    with open('market_caps.json', 'r') as f:
        market_caps = json.load(f)
    # Load common stocks
    with open('responses/common_stocks.json', 'r') as f:
        common_stocks = json.load(f)
    # remove 'J' and 'TSCO' from common_stocks and market_caps
    common_stocks = [stock for stock in common_stocks if stock not in ['J', 'TSCO']]
    market_caps = {k: v for k, v in market_caps.items() if k not in ['J', 'TSCO']}

    # Load returns data
    returns = pd.read_csv(f'yfinance/returns_{start_date}_{end_date}.csv', index_col=0)
    returns = returns[returns.columns.intersection(market_caps.keys())]
    returns = returns[returns.columns.intersection(common_stocks)]
    nan_cols = returns.columns[returns.isna().any()]
    returns = returns.dropna(axis=1)
    sp500_tickers = returns.columns.tolist()

    # Calculate market cap weighted returns
    market_caps_series = pd.Series(market_caps)
    valid_tickers = market_caps_series.dropna().index
    market_cap_weights = market_caps_series.loc[valid_tickers] / market_caps_series.loc[valid_tickers].sum()
    market_return_weighted = (returns[valid_tickers] * market_cap_weights).sum(axis=1)

    # Calculate market equilibrium returns
    risk_free_rate = 0.02
    market_var = market_return_weighted.var()
    market_beta = returns[valid_tickers].apply(lambda x: x.cov(market_return_weighted)) / market_var
    market_risk_premium = (market_return_weighted - risk_free_rate).mean()
    market_equilibrium_return = market_beta * market_risk_premium

    # Load LLM responses
    # with open(f'responses/llama_{start_date}_{end_date}.json', 'r') as f:
    #     llama_dict = json.load(f)
    # with open(f'responses/qwen_{start_date}_{end_date}.json', 'r') as f:
    #     qwen_dict = json.load(f)
    with open(f'responses/gemma_{start_date}_{end_date}.json', 'r') as f:
        gemma_dict = json.load(f)

    # Remove nan columns and stocks not in returns data
    # llama_dict = {k: v for k, v in llama_dict.items() if k not in nan_cols}
    # qwen_dict = {k: v for k, v in qwen_dict.items() if k not in nan_cols}
    gemma_dict = {k: v for k, v in gemma_dict.items() if k not in nan_cols}

    # Calculate Black-Litterman results
    # black_litterman_llama = black_litterman_LLM(llama_dict, returns, sp500_tickers, market_equilibrium_return, tau)
    # black_litterman_qwen = black_litterman_LLM(qwen_dict, returns, sp500_tickers, market_equilibrium_return, tau)
    black_litterman_gemma = black_litterman_LLM(gemma_dict, returns, sp500_tickers, market_equilibrium_return, tau)

    # return pd.Series(black_litterman_llama, index=sp500_tickers), pd.Series(black_litterman_qwen, index=sp500_tickers), pd.Series(black_litterman_gemma, index=sp500_tickers)
    return pd.Series(black_litterman_gemma, index=sp500_tickers)

def main():
    tau = 0.025  # Set tau as hyperparameter
    
    date_pairs = [
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-07-31"),
        ("2024-08-01", "2024-08-31"),
        ("2024-09-01", "2024-09-30"),
        ("2024-10-01", "2024-10-31"),
        ("2024-11-01", "2024-11-30"),
        ("2024-12-01", "2024-12-31")
    ]

    # llama_results = {}
    # qwen_results = {}
    gemma_results = {}

    for start_date, end_date in tqdm(date_pairs):
        print(f"Processing period: {start_date} to {end_date}")
        try:
            # llama_weights, qwen_weights, gemma_weights = process_period(start_date, end_date, tau)
            gemma_weights = process_period(start_date, end_date, tau)
            # llama_results[(start_date, end_date)] = llama_weights
            # qwen_results[(start_date, end_date)] = qwen_weights
            gemma_results[(start_date, end_date)] = gemma_weights
        except Exception as e:
            print(f"Error processing period {start_date} to {end_date}: {str(e)}")

    # Convert results to DataFrames
    # llama_results_df = pd.DataFrame(llama_results).T
    # qwen_results_df = pd.DataFrame(qwen_results).T
    gemma_results_df = pd.DataFrame(gemma_results).T

    # Convert MultiIndex to a single Date column (using just the start_date)
    # llama_results_df = llama_results_df.reset_index()
    # qwen_results_df = qwen_results_df.reset_index()
    gemma_results_df = gemma_results_df.reset_index()
    
    # Take only the start_date (level_0) as the Date
    # llama_results_df['Date'] = llama_results_df['level_0']
    # qwen_results_df['Date'] = qwen_results_df['level_0']
    gemma_results_df['Date'] = gemma_results_df['level_0']
    
    # Drop the original index columns
    # llama_results_df = llama_results_df.drop(['level_0', 'level_1'], axis=1)
    # qwen_results_df = qwen_results_df.drop(['level_0', 'level_1'], axis=1)
    gemma_results_df = gemma_results_df.drop(['level_0', 'level_1'], axis=1)

    # Save results with tau value in filenames
    # llama_results_df.to_csv(f'results/llama_black_litterman_weights_tau_{tau}.csv', index=False)
    # qwen_results_df.to_csv(f'results/qwen_black_litterman_weights_tau_{tau}.csv', index=False)
    gemma_results_df.to_csv(f'results/gemma_black_litterman_weights_tau_{tau}.csv', index=False)

    print("\nResults shape:")
    # print(f"LLAMA results: {llama_results_df.shape}")
    # print(f"Qwen results: {qwen_results_df.shape}")
    print(f"Gemma results: {gemma_results_df.shape}")

if __name__ == "__main__":
    main()