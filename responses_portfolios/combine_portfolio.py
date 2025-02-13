import pandas as pd
import glob
import os

def combine_portfolios(pattern, output_file):
    # Get current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all portfolio files matching the pattern
    files = sorted(glob.glob(os.path.join(current_dir, pattern)))
    print(files)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
        
    # Read and combine all files
    dfs = []
    for file in sorted(files):
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, axis=0)
    
    # Sort by date if there's a date column
    if 'date' in combined_df.columns:
        combined_df = combined_df.sort_values('date')
    
    # Save the combined file
    output_path = os.path.join(current_dir, output_file)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined portfolio saved as '{output_file}'")

# Combine equal weighted portfolios
combine_portfolios('equal_weighted_portfolio_*.csv', 'equal_weighted_portfolio.csv')

# Combine optimized portfolios
combine_portfolios('optimized_portfolio_*.csv', 'optimized_portfolio.csv')
