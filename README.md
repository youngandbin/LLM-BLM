# README: Stock Return Prediction using LLMs

## Overview

This project predicts stock returns using a Language Model (LLM). Given historical stock price data from **Yahoo Finance**, the model predicts the **average daily return** for the next month. The project supports multiple models, including:

- **GPT-4o-mini**
- **LLaMA 3.1 (8B)**
- **Gemma 7B**

The predictions are generated using an LLM-based approach, which takes **past stock returns and company metadata** as input.

## Features

- Fetches **S&P 500** stock data from **Yahoo Finance**.
- Uses **Pandas** to process stock return data.
- Queries a **Language Model** to predict **next-month returns**.
- Saves results as **JSON** for analysis.

## Setup

### 1. Install Dependencies

```sh
pip install numpy pandas yfinance tqdm openai pydantic argparse
```

### 2. Set Up API Keys

- If using **GPT-4o-mini**, replace the OpenAI API key.
- If using **LLaMA** or **Gemma**, update the `base_url` for your model API.

## Usage

### Run Prediction

```sh
python script.py --model_name gpt
```

Replace `gpt` with `llama` or `gemma` as needed.

### Output Files

- **Stock Data:** `yfinance/data_YYYY-MM-DD_YYYY-MM-DD.csv`
- **Returns Data:** `yfinance/returns_YYYY-MM-DD_YYYY-MM-DD.csv`
- **Predictions:** `responses/{model_name}_YYYY-MM-DD_YYYY-MM-DD.json`

## Prediction Process

1. **Fetch S&P 500 stock list** from Wikipedia.
2. **Download daily close prices** for each stock.
3. **Compute daily percentage change** for returns.
4. **Format the data** into prompts for the LLM.
5. **Query the model 30 times per stock** for variance analysis.
6. **Save predictions** to a JSON file.

## Example Response Format

```json
{
    "AAPL": {
        "ticker": "AAPL",
        "Security": "Apple Inc.",
        "GICS Sector": "Information Technology",
        "GICS Sub-Industry": "Technology Hardware",
        "pct_change": [0.01, -0.005, 0.02, ...],
        "expected_return": [0.003, 0.002, 0.004, ...]
    }
}
```

## Notes

- This project does **not** provide financial advice.
- Ensure your **API key** is valid when using **OpenAI**.
- Modify `base_url` for **local LLaMA/Gemma models**.

## License

MIT License. Feel free to modify and use. 🚀
