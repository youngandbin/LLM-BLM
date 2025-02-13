"""
this code makes two different portfolios with the start_date and end_date.  
wanna make it monthly, with [2024-06-01~2024-06-30 to 2024-07-01~2024-07-31], ..., [2024-11-01~2024-11-30 to 2024-12-01~2024-12-31]. That is, you should create each portfolios 6 times from June 2024 to Nov 2024.
"""

import pandas as pd
from scipy.optimize import minimize
import numpy as np
from datetime import datetime, timedelta
import calendar
import json

def get_last_day_of_month(year, month):
    return calendar.monthrange(year, month)[1]

# Months to process (June to December 2024)
months = range(6, 13)  # 6 = June, 12 = December

for month in months:
    
    """
    Data 
    """
    # Set up dates for current month (training) and next month (testing)
    train_year = 2024
    train_month = month
    
    # Handle year transition for December to January
    if month == 12:
        test_month = 1
        test_year = 2025
    else:
        test_month = month + 1
        test_year = train_year
    
    train_start = f'2024-{train_month:02d}-01'
    train_end = f'2024-{train_month:02d}-{get_last_day_of_month(2024, train_month):02d}'
    test_start = f'{test_year}-{test_month:02d}-01'
    test_end = f'{test_year}-{test_month:02d}-{get_last_day_of_month(test_year, test_month):02d}'
    
    print(f"\nProcessing: Training {train_start} to {train_end}, Testing {test_start} to {test_end}")
    
    # Read training and testing data
    data = pd.read_csv(f'yfinance/returns_{train_start}_{train_end}.csv')
    future_data = pd.read_csv(f'yfinance/returns_{test_start}_{test_end}.csv')

    """
    Data preprocessing
    """
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
    date_col = data['Date']  # Save Date column
    future_date_col = future_data['Date']  # Save Date column
    data = data[data.columns.intersection(market_caps.keys())]
    future_data = future_data[future_data.columns.intersection(common_stocks)]
    data = data.dropna(axis=1)
    future_data = future_data.dropna(axis=1)
    # Restore Date columns
    data['Date'] = date_col
    future_data['Date'] = future_date_col

    
    """
    equal weighted portfolio
    """
    # Get asset columns (excluding Date)
    asset_columns = [col for col in data.columns if col != 'Date']
    n_assets = len(asset_columns)
    
    # Equal weighted portfolio
    weight = 1/n_assets
    equal_weighted_returns = (future_data[asset_columns] * weight).sum(axis=1)
    equal_weighted_portfolio = pd.DataFrame({
        'Date': future_data['Date'],
        'Portfolio_Return': equal_weighted_returns
    })
    
    # Save equal weighted portfolio results
    equal_weighted_portfolio.to_csv(f'responses_portfolios/equal_weighted_portfolio_{train_start}_{train_end}.csv', index=False)
    
    """
    optimized portfolio
    """
    # Mean-variance optimized portfolio
    mean_returns = data[asset_columns].mean()
    cov_matrix = data[asset_columns].cov()
    lambda_param = 0.1
    
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_risk - (lambda_param * portfolio_return)
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Fully invested portfolio constraint
        {'type': 'ineq', 'fun': lambda x: x}  # No short-selling constraint (weights must be >= 0)
    )

    bounds = tuple((0,1) for asset in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_weights, 
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    optimal_weights = result.x
    optimized_returns = (future_data[asset_columns] * optimal_weights).sum(axis=1)
    
    optimized_portfolio = pd.DataFrame({
        'Date': future_data['Date'],
        'Portfolio_Return': optimized_returns
    })
    
    # Save optimized portfolio results
    optimized_portfolio.to_csv(f'responses_portfolios/optimized_portfolio_{train_start}_{train_end}.csv', index=False)
    
    print(f"Saved portfolio results for training period: {train_start} to {train_end}") 