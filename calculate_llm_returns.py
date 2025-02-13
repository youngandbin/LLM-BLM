import pandas as pd
import numpy as np
import calendar
from datetime import datetime

def get_last_day_of_month(year, month):
    return calendar.monthrange(year, month)[1]

def calculate_llm_returns(tau = 0.025):
    # Read LLM results
    llama_results = pd.read_csv(f'results/llama_black_litterman_weights_tau_{tau}.csv')
    qwen_results = pd.read_csv(f'results/qwen_black_litterman_weights_tau_{tau}.csv')
    gemma_results = pd.read_csv(f'results/gemma_black_litterman_weights_tau_{tau}.csv')

    llama_monthly_returns = []
    qwen_monthly_returns = []
    gemma_monthly_returns = []

    # Process each month from June to November 2024 (weights)
    for month in range(6, 12):  # Now only going up to November since December weights will be for January
        # The weights are for this month
        weight_date = f'2024-{month:02d}-01'
        
        # The returns are for next month
        returns_month = month + 1
        returns_year = 2024
        returns_start = f'{returns_year}-{returns_month:02d}-01'
        returns_end = f'{returns_year}-{returns_month:02d}-{get_last_day_of_month(returns_year, returns_month):02d}'
        
        # Read returns data for the next month
        future_data = pd.read_csv(f'yfinance/returns_{returns_start}_{returns_end}.csv')
        
        # Get asset columns (excluding Date column)
        asset_columns = [col for col in llama_results.columns if col != 'Date']
        
        # Get weights for the current month
        llama_month_weights = llama_results[llama_results['Date'] == weight_date][asset_columns].iloc[0].values
        qwen_month_weights = qwen_results[qwen_results['Date'] == weight_date][asset_columns].iloc[0].values
        gemma_month_weights = gemma_results[gemma_results['Date'] == weight_date][asset_columns].iloc[0].values
        
        # Calculate portfolio returns
        llama_returns = (future_data[asset_columns] * llama_month_weights).sum(axis=1)
        qwen_returns = (future_data[asset_columns] * qwen_month_weights).sum(axis=1)
        gemma_returns = (future_data[asset_columns] * gemma_month_weights).sum(axis=1)
        
        # Create DataFrames with dates
        llama_portfolio = pd.DataFrame({
            'Date': future_data['Date'],
            'Portfolio_Return': llama_returns
        })
        
        qwen_portfolio = pd.DataFrame({
            'Date': future_data['Date'],
            'Portfolio_Return': qwen_returns
        })

        gemma_portfolio = pd.DataFrame({
            'Date': future_data['Date'],
            'Portfolio_Return': gemma_returns
        })
        
        llama_monthly_returns.append(llama_portfolio)
        qwen_monthly_returns.append(qwen_portfolio)
        gemma_monthly_returns.append(gemma_portfolio)

    # Combine all monthly returns
    llama_all_returns = pd.concat(llama_monthly_returns, ignore_index=True)
    qwen_all_returns = pd.concat(qwen_monthly_returns, ignore_index=True)
    gemma_all_returns = pd.concat(gemma_monthly_returns, ignore_index=True)
    
    return llama_all_returns, qwen_all_returns, gemma_all_returns

if __name__ == "__main__":
    tau = 0.025
    llama_returns, qwen_returns, gemma_returns = calculate_llm_returns(tau = tau)
    print("LLaMA Portfolio Returns:")
    print(llama_returns.head())
    print("\nQwen Portfolio Returns:")
    print(qwen_returns.head())
    print("\nGemma Portfolio Returns:")
    print(gemma_returns.head())
    # save to csv
    llama_returns.to_csv(f'results/llama_black_litterman_returns_tau_{tau}.csv', index=False)
    qwen_returns.to_csv(f'results/qwen_black_litterman_returns_tau_{tau}.csv', index=False)
    gemma_returns.to_csv(f'results/gemma_black_litterman_returns_tau_{tau}.csv', index=False)