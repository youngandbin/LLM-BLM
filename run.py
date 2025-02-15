import numpy as np
import pandas as pd
import yfinance as yf
# from scipy.optimize import minimize
from tqdm import tqdm
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta
class ResearchPaperExtraction(BaseModel):
    expected_return: float

# make model_name arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama")
args = parser.parse_args()

"""
model
"""
model_name = args.model_name
if model_name == 'llama':
    client = OpenAI(
        base_url=f"https://{YOUR_API_ENDPOINT}", 
        api_key="-",
    )
elif model_name == 'gemma':
    client = OpenAI(
        base_url=f"https://{YOUR_API_ENDPOINT}",
        api_key="-",
    )
elif model_name == 'qwen':
    client = OpenAI(
        base_url=f"https://{YOUR_API_ENDPOINT}", 
        api_key="-",
    )

"""
data
"""

# Create date range from June 2024 to November 2024
date_range = pd.date_range(start="2024-06-01", end="2024-12-01", freq='MS')

# Create directories if they don't exist
os.makedirs('yfinance', exist_ok=True)
os.makedirs('responses', exist_ok=True)

# 1. S&P500 ticker data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)[0]  # Wikipedia table data
sp500_table = tables[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

for current_date in tqdm(date_range):
    month_start = current_date.strftime('%Y-%m-%d')
    month_end = (current_date + pd.DateOffset(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Processing data for {month_start} - {month_end}")
    
    # Download data using yfinance
    data = yf.download(sp500_table['Symbol'].tolist(), start=month_start, end=month_end)['Close']
    # Save raw data
    data.to_csv(f'yfinance/data_{month_start}_{month_end}.csv')
    
    # Process returns
    returns = data.pct_change().iloc[1:]
    returns.to_csv(f'yfinance/returns_{month_start}_{month_end}.csv')
    sp500_tickers = returns.columns

    # Organize data
    data_dict = {}
    for ticker in sp500_tickers:
        data_dict[ticker] = {
            'ticker': ticker,
            'Security': sp500_table[sp500_table['Symbol'] == ticker]['Security'].values[0],
            'GICS Sector': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0],
            'GICS Sub-Industry': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0],
            'pct_change': returns[ticker].tolist()
        }

    def make_system_prompt():
        return """You are a language model designed to predict stock returns. Given a time-series of daily returns as percentage change from the past month, along with the following company information in a dictionary:

        - symbol: The stock symbol
        - company name: The name of the company
        - GICS sector: The Global Industry Classification Standard (GICS) sector
        - GICS sub-industry: The GICS sub-industry

        You should predict the average daily return as a percentage change for the next month. Return a single float value representing the predicted average daily return for the next month. Do not include any additional commentary, explanations, or information. Only output the float value.
        """

    def make_user_prompt(ticker, data_dict):
        return f"""Ticker: {ticker}
        Security: {data_dict[ticker]['Security']}
        Sector: {data_dict[ticker]['GICS Sector']}
        Sub-Industry: {data_dict[ticker]['GICS Sub-Industry']}
        pct_change: {data_dict[ticker]['pct_change']}"""

    for ticker in tqdm(sp500_tickers):
        print("="*80)
        print(ticker)
        print("="*80)
        security = data_dict[ticker]['Security']
        sector = data_dict[ticker]['GICS Sector']
        sub_industry = data_dict[ticker]['GICS Sub-Industry']
        pct_change = data_dict[ticker]['pct_change']
        system_prompt = make_system_prompt()
        user_prompt = make_user_prompt(ticker, data_dict)
        if model_name == 'gemma':   
            # concat system prompt and user prompt (cuz gemma does not support system prompt)
            user_prompt = system_prompt + "\n\n" + user_prompt

        # Get responses 30 times, and use the variance of those values as the confidence of the view.
        answers = []
        for _ in range(30):
            if model_name == 'llama':
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            elif model_name == 'gemma':
                completion = client.chat.completions.create(
                    model="google/gemma-7b-it",
                    messages=[
                        # {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            elif model_name == 'qwen':
                completion = client.chat.completions.create(
                    model="qwen2-7b-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            try:
                expected_return_dict = eval(completion.choices[0].message.content)
                print(expected_return_dict)
            except:
                # print(completion.choices[0].message.content)
                continue
            answers.append(expected_return_dict['expected_return'])
        data_dict[ticker]['expected_return'] = answers

    # Save responses for this month
    with open(f'responses/{model_name}_{month_start}_{month_end}.json', 'w') as f:
        json.dump(data_dict, f)
