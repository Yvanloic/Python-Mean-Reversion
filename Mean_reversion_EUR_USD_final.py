# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 01:46:40 2025

@author: DJOKO Yvan , EYOUM Ingrid

"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations

#Part A - Finding alpha and beta for the mean reversion

#Q1)

# Download historical data for EUR/USD
# 'EURUSD=X' is the ticker symbol for the EUR/USD exchange rate
data = yf.download('EURUSD=X', 
                   start='2010-01-01',  # Start date: January 1, 2010
                   end='2024-12-31',    # End date: December 31, 2024
                   interval='1d')       # Daily frequency


if isinstance(data.columns, pd.MultiIndex):  # Check if the dataframe is a multiIndex
    data.columns = data.columns.get_level_values(0)

# Save the data to a CSV file
#data.to_csv('eur_usd_data.csv')

#check if we have nan values
num_nan_rows = data.isna().any(axis=1).sum()
print(f"Number of rows with NaN values: {num_nan_rows}")

print(data.columns)
print(data.head)
print(f"columns in the dataset: {data.columns}")

#Data Cleaning

# Drop the 'Volume' column
data = data.drop(columns=['Adj Close', 'Volume'], errors = 'ignore')

# Rename the columns for simplicity
data.rename(columns = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace = True)

# Change the index to datetime
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
data.set_index('Date', inplace=True)
print(data.head)

# Ensure the dataset contains only Monday-Friday data which are trading days
data = data[data.index.dayofweek < 5]
# Check if any weekends are present (should be 0)
print(f"Number of weekend dates in dataset: {(data.index.dayofweek >= 5).sum()}")
#data.head()
#data.to_excel( r"C:\Users\2103020\Downloads\data.xlsx", index = True) #if we want to extract the data

# EDA: Visualizing Trends (Line plot of exchange rates over time)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='EUR/USD Close Price', color='blue')
plt.title('EUR/USD Exchange Rate (Close) Over Time')
plt.xlabel('Date')
plt.ylabel('EUR/USD Exchange Rate')
plt.legend()
plt.grid(True)
plt.show()

# Compute Summary Statistics (Mean, Variance)
mean_close = data['Close'].mean()  # Mean of 'Close' price
variance_close = data['Close'].var()  # Variance of 'Close' price
mean_open = data['Open'].mean()  # Mean of 'Open' price
variance_open = data['Open'].var()  # Variance of 'Open' price

print(f"Mean of Close Price: {mean_close}")
print(f"Variance of Close Price: {variance_close}")
print(f"Mean of Open Price: {mean_open}")
print(f"Variance of Open Price: {variance_open}")

# Visualize some more aspects (optional): High and Low prices over time
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['High'], label='High Price', color='green')
plt.plot(data.index, data['Low'], label='Low Price', color='red')
plt.title('EUR/USD High and Low Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


#  Mean reversion & Time-Series Regression
# Calculate daily changes (delta of Close price)
data['Delta_Close'] = data['Close'].diff()
# Preview the data to make sure it's correct
data.head()

#before running OLS regression always check for stationarity to avoid spurious results and ensure reliable inference
#ADF test for stationarity
result = adfuller(data['Delta_Close'].dropna())  
if result[1] < 0.05:
    print("Serie of daily changes is stationary(p-value < 0.05).")
else:
    print("Serie of daily changes is NOT stationary (p-value >= 0.05).")


# Define the lagged variable (X_t-1)
data['Lagged_Close'] = data['Close'].shift(1)
data.head()

# Drop the first row due to NaN values from lagging
data = data.dropna(subset=['Lagged_Close'])
#print(data.head())

# Prepare the regression variables (dependent and independent variables)
X = data['Lagged_Close']  # Independent variable (X_t-1)
y = data['Delta_Close']   # Dependent variable (Delta X_t)

# Add constant (intercept) to the model
X = sm.add_constant(X)

# Perform OLS regression
model = sm.OLS(y, X).fit()

# Display the results of the regression
print(model.summary())


#Check Mean Reversion

# Extract the alpha and beta values and their t-statistics
alpha = model.params['const']
beta = model.params['Lagged_Close']
t_stat_beta = model.tvalues['Lagged_Close']

# Display the results
print(f"Alpha: {alpha}")
print(f"Beta: {beta}")
print(f"T-statistic for Beta: {t_stat_beta}")

# Check for mean reversion feature
if alpha > 0 and beta < 0 and t_stat_beta < -1: 
    print("Mean reversion feature is present.")
else:
    print("No mean reversion feature detected.")


#Visualize results

# Calculate long-run mean
long_run_mean = -alpha / beta
# Print the calculated long-run mean
long_run_mean = -alpha / beta
print(f"Long-run mean: {long_run_mean}")

# Create a plot of Close prices vs. the long-run mean
plt.figure(figsize=(10, 6))
# Plot the actual Close prices
plt.plot(data.index, data['Close'], label='EUR/USD Close Price', color='blue')
# Plot the long-run mean (constant line)
plt.axhline(long_run_mean, color='red', linestyle='--', label=f'Long-run Mean: {long_run_mean:.4f}')
# Add titles and labels
plt.title('EUR/USD Close Price vs. Long-run Mean')
plt.xlabel('Date')
plt.ylabel('EUR/USD Exchange Rate')
plt.legend()
plt.grid(True)
# Show plot
plt.show()


#Q2)  Let's test therefore with a rolling window period.

# Rolling window analysis (500-day rolling windows)
window_size = 500
rolling_results = []

for start in range(len(data) - window_size):
    subset = data.iloc[start:start + window_size]
    
    X_sub = subset[['Lagged_Close']]
    y_sub = subset['Delta_Close']
    
    X_sub = sm.add_constant(X_sub)
    
    model_sub = sm.OLS(y_sub, X_sub).fit()
    alpha_sub, beta_sub = model_sub.params
    t_stat_beta_sub = model_sub.tvalues['Lagged_Close']
    
    rolling_results.append([
        subset.index[-1], alpha_sub, beta_sub, t_stat_beta_sub,
        (alpha_sub > 0) and (beta_sub < 0) and (t_stat_beta_sub < -1)
    ])

rolling_results_df = pd.DataFrame(rolling_results, columns=['Date', 'Alpha', 'Beta', 't-stat(Beta)', 'Mean Reversion'])

# Visualize Rolling Mean Reversion Over Time to see periods with strong mean reversion features. 
plt.figure(figsize=(12, 6))
plt.plot(rolling_results_df['Date'], rolling_results_df['Beta'], label='Rolling Beta', color='blue')
plt.axhline(0, linestyle='--', color='red', label='Zero Line')
plt.title('Rolling Beta Over Time')
plt.xlabel('Date')
plt.ylabel('Beta Coefficient')
plt.legend()
plt.grid(True)
plt.show()

#FUNCTIONT  TO TEST MEAN REVERSION ACROSS OTHER CURRENCIES (EUR/GBP, GBP/USD, JPY/USD)

currency_pairs = ["EURUSD=X", "EURGBP=X", "GBPUSD=X", "JPYUSD=X"]  # You can add more
start_date = "2010-01-01"
end_date = "2024-12-31"

# Download forex data from Yahoo Finance
df = yf.download(currency_pairs, start=start_date, end=end_date, interval="1d")['Close']
#df.to_csv("Project_Database.csv")

# Function to test mean reversion properties
def test_mean_reversion(data, currency_pair):
    """
    Tests mean reversion properties of a forex time series using OLS regression,
    calculating alpha, beta, t-statistic, and performing an ADF test.

    Parameters:
        data (pd.Series): Time series of currency price
        currency_pair (str): Name of the currency pair
    
    Returns:
        results (dict): Mean reversion test results
    """
    # Compute price differences
    data = data.dropna()
    price_changes = data.diff().dropna()

    # Lagged price as independent variable
    X = sm.add_constant(data.shift(1).dropna())
    y = price_changes[X.index]  # Ensure matching index

    # OLS regression: ΔX_t = α + β * X_t-1 + ε_t
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params
    t_stat = model.tvalues[1]

    # Augmented Dickey-Fuller test for stationarity
    adf_result = adfuller(data, autolag="AIC")
    adf_stat, p_value = adf_result[0], adf_result[1]

    # Output results
    results = {
        "Currency Pair": currency_pair,
        "Alpha": alpha,
        "Beta": beta,
        "t-Statistic": t_stat,
        "ADF Statistic": adf_stat,
        "ADF p-value": p_value,
    }
    
    return results

# Run tests for each currency pair
results_list = []

for pair in currency_pairs:
    results = test_mean_reversion(df[pair], pair)
    results_list.append(results)

# Convert results into DataFrame and print
results_df = pd.DataFrame(results_list)
print(results_df)


#Part B : Trading strategy implementation

#We implemented the strategy through a function, so it'll be easy to do it for other currencies. 

def strategy_mean_reversion(ticker, start_date, end_date, estimation_window=80):
    """
    Implements the Mean Reversion strategy for a given currency pair.

    Parameters:
    - ticker : String representing the currency pair ticker (e.g., "EURUSD=X").
    - start_date : String representing the start date (format: "YYYY-MM-DD").
    - end_date : String representing the end date (format: "YYYY-MM-DD").
    - estimation_window : Rolling regression window size (default is 80 days).

    Returns:
    - Displays performance plots and prints performance metrics.
    """
    # Download data with user-defined dates
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if isinstance(data.columns, pd.MultiIndex):  # Check if the dataframe is a multiIndex
        data.columns = data.columns.get_level_values(0)

    # Data cleaning
    data.drop(columns=['Adj Close', 'Volume'], errors='ignore', inplace=True)
    data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.set_index('Date', inplace=True)

    # Create necessary columns
    data['Delta_Close'] = data['Close'].diff()
    data['Lagged_Close'] = data['Close'].shift(1)
    data.dropna(subset=['Lagged_Close'], inplace=True)

    # Initialize columns
    data['Alpha'] = np.nan
    data['Beta'] = np.nan
    data['t_stat_Beta'] = np.nan
    data['Predicted_Change'] = np.nan
    data['Signal'] = 0

    # Rolling regression
    for start in range(len(data) - estimation_window):
        subset = data.iloc[start:start + estimation_window]
        X_sub = sm.add_constant(subset[['Lagged_Close']])
        y_sub = subset['Delta_Close']

        model = sm.OLS(y_sub, X_sub).fit()
        alpha_hat, beta_hat = model.params
        t_stat_beta_hat = model.tvalues['Lagged_Close']

        next_index = start + estimation_window
        current_price = data.iloc[next_index]['Close']
        data.at[data.index[next_index], 'Alpha'] = alpha_hat
        data.at[data.index[next_index], 'Beta'] = beta_hat
        data.at[data.index[next_index], 't_stat_Beta'] = t_stat_beta_hat
        data.at[data.index[next_index], 'Predicted_Change'] = alpha_hat + beta_hat * current_price

        # Generate buy and sell signals
        if alpha_hat > 0 and beta_hat < 0 and t_stat_beta_hat < -1:
            if data.at[data.index[next_index], 'Predicted_Change'] > 0:
                data.at[data.index[next_index], 'Signal'] = 1  # Buy
            elif data.at[data.index[next_index], 'Predicted_Change'] < 0:
                data.at[data.index[next_index], 'Signal'] = -1  # Sell

    # Calculate strategy returns
    data['Strat_Daily_Return'] = data['Signal'].shift(1) * data['Close'].pct_change()
    data['Strat_Cumulative_Return'] = (1 + data['Strat_Daily_Return']).cumprod()
    data['Buy_and_Hold'] = (1 + data['Close'].pct_change()).cumprod()

    # Compute performance metrics
    sharpe_ratio = (data['Strat_Daily_Return'].mean() / data['Strat_Daily_Return'].std()) * np.sqrt(252)
    rolling_max = data['Strat_Cumulative_Return'].cummax()
    drawdown = (rolling_max - data['Strat_Cumulative_Return']) / rolling_max
    max_drawdown = drawdown.max()

    # Plot performance
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Strat_Cumulative_Return'], label=f'Mean Reversion Strategy ({ticker})', color='blue', linewidth=1.5)
    plt.plot(data.index, data['Buy_and_Hold'], label=f'Buy & Hold Strategy ({ticker})', linestyle='dashed', color='black')
    plt.scatter(data.index[data['Signal'] == 1], data['Strat_Cumulative_Return'][data['Signal'] == 1], label='Buy Signal', marker='^', color='Lime', edgecolors='black', s=80, alpha=0.9)
    plt.scatter(data.index[data['Signal'] == -1], data['Strat_Cumulative_Return'][data['Signal'] == -1], label='Sell Signal', marker='v', color='red', edgecolors='black', s=80, alpha=0.9)
    plt.title(f'Mean Reversion Strategy Performance ({ticker})', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

    # Print performance metrics
    print(f"Sharpe Ratio ({ticker}): {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown ({ticker}): {max_drawdown:.2%}")
    print(f"Final Return - Mean Reversion Strategy ({ticker}): {data['Strat_Cumulative_Return'].iloc[-1]:.2f}")
    print(f"Final Return - Buy & Hold Strategy ({ticker}): {data['Buy_and_Hold'].iloc[-1]:.2f}")

#Usage
strategy_mean_reversion("EURUSD=X", start_date="2010-01-01", end_date="2024-12-31")


#PART B: 
#Q3) Let's Refine the previous strategy

# Refinement 01 : Hybrid Lookback Window Mean-Reversion Strategy

def mean_reversion_hybrid_lookback(ticker, start_date, end_date, currency_pair=None):
    """
    Implements the Hybrid Lookback Window Mean-Reversion Strategy for a given forex pair.

    Parameters:
        ticker (str): The Yahoo Finance ticker symbol for the currency pair (e.g., "EURUSD=X").
        start_date (str): Start date for data download (format: "YYYY-MM-DD").
        end_date (str): End date for data download (format: "YYYY-MM-DD").
        currency_pair (str, optional): The name of the forex pair for display (default: inferred from ticker).
    
    Returns:
        performance_hybrid (pd.DataFrame): DataFrame with cumulative performance results.
        metrics_hybrid (pd.DataFrame): DataFrame with strategy performance metrics.
    """

    # Step 1: Download historical forex data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    if isinstance(df.columns, pd.MultiIndex):  # Handle MultiIndex if present
        df.columns = df.columns.get_level_values(0)

    # Data cleaning
    df.drop(columns=['Adj Close', 'Volume'], errors='ignore', inplace=True)
    df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Set default currency pair name if not provided
    if currency_pair is None:
        currency_pair = ticker.replace("=X", "")

    # Step 2: Compute rolling volatility (Standard deviation of returns over 30 days)
    df['Volatility'] = df['Close'].pct_change().rolling(window=30).std()

    # Step 3: Initialize capital and performance tracking
    capital = 1  # Start with 1 USD
    capital_series = []
    returns = []
    positions = []

    # Step 4: Implement the Hybrid Lookback Mean-Reversion Strategy
    for i in range(100, len(df) - 1):  
        # Define hybrid lookback window based on volatility
        volatility = df['Volatility'].iloc[i]
        if volatility > df['Volatility'].quantile(0.75):  
            lookback = 70  
        elif volatility < df['Volatility'].quantile(0.25):  
            lookback = 100  
        else:
            lookback = 80  

        # Ensure sufficient data for regression
        if i - lookback < 0:
            continue

        rolling_df = df.iloc[i - lookback:i]

        # Define independent (X_t-1) and dependent (ΔX) variables
        X_rolling = rolling_df['Close'].shift(1).dropna()
        y_rolling = rolling_df['Close'].diff().dropna()
        X_rolling = sm.add_constant(X_rolling)

        # Fit regression model
        model_rolling = sm.OLS(y_rolling, X_rolling).fit()
        alpha_hat, beta_hat = model_rolling.params
        t_beta_hat = model_rolling.tvalues[1]

        # Predict next day's price movement
        X_t = df.iloc[i]['Close']
        expected_change = alpha_hat + beta_hat * X_t

        # Trading decision based on mean reversion conditions
        if alpha_hat > 0 and beta_hat < 0 and t_beta_hat < -1:
            position = 1 if expected_change > 0 else -1  
        else:
            position = 0  

        positions.append(position)

        # Compute return based on position
        next_day_return = (df.iloc[i + 1]['Close'] - X_t) / X_t
        strategy_return = position * next_day_return
        returns.append(strategy_return)

        # Update capital
        capital *= (1 + strategy_return)
        capital_series.append(capital)

    # Step 5: Convert results into a DataFrame
    performance_hybrid = pd.DataFrame({
        'Date': df.index[100:len(df) - 1],
        'Capital': capital_series
    }).set_index('Date')

    # Step 6: Compute Performance Metrics
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = (performance_hybrid['Capital'].min() / performance_hybrid['Capital'].max()) - 1
    cumulative_return = performance_hybrid['Capital'].iloc[-1]  

    metrics_hybrid = pd.DataFrame({
        "Strategy": ["Hybrid Lookback Mean-Reversion"],
        "Currency Pair": [currency_pair],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown": [max_drawdown],
        "Cumulative Return ": [cumulative_return]
    })

    # Step 7: Plot cumulative performance
    plt.figure(figsize=(12, 6))
    plt.plot(performance_hybrid['Capital'], label=f"{currency_pair} Hybrid Lookback Strategy", color="blue")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Cumulative Performance of Hybrid Lookback Mean-Reversion Strategy ({currency_pair})")
    plt.xlabel("Date")
    plt.ylabel("Capital (Starting at 1 USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Step 8: Print Performance Metrics
    print("\nStrategy Performance Metrics:")
    print(metrics_hybrid.to_string(index=False))

    return performance_hybrid, metrics_hybrid

# Usage:
performance_hybrid, metrics_hybrid = mean_reversion_hybrid_lookback(
    ticker="EURUSD=X",
    start_date="2010-01-01",
    end_date="2024-12-31"
)


#Refinement 02: mean_reversion_momentum_filter

def mean_reversion_momentum_filter(ticker, start_date, end_date, currency_pair=None):
    """
    Implements the Mean-Reversion Strategy with Momentum Filter for a given forex pair.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol (e.g., "EURUSD=X").
        start_date (str): Start date for historical data download (format: "YYYY-MM-DD").
        end_date (str): End date for historical data download (format: "YYYY-MM-DD").
        currency_pair (str): The name of the forex pair (default: "EUR/USD").
    
    Returns:
        performance_momentum (pd.DataFrame): DataFrame with cumulative performance results.
        metrics_momentum (pd.DataFrame): DataFrame with strategy performance metrics.
    """

    # Step 1: Download Forex Data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if isinstance(data.columns, pd.MultiIndex):  # Check if the dataframe is a multiIndex
        data.columns = data.columns.get_level_values(0)

    # Data cleaning
    data = data[['Close']].dropna()
    
    # Compute rolling volatility (Standard deviation of returns over 30 days)
    data['Volatility'] = data['Close'].pct_change().rolling(window=30).std()

    # Compute 10-day simple moving average (SMA) for momentum filtering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()

    # Initialize capital and performance tracking
    capital = 1  # Start with 1 USD
    capital_series = []
    returns = []
    positions = []

    # Implement the Mean-Reversion Strategy with Momentum Filter
    for i in range(100, len(data) - 1):  # Start from 100 to allow for lookback window
        # Define hybrid lookback window based on volatility
        volatility = data['Volatility'].iloc[i]
        if volatility > data['Volatility'].quantile(0.75):  
            lookback = 70  # High volatility → Shorter window
        elif volatility < data['Volatility'].quantile(0.25):  
            lookback = 100  # Low volatility → Longer window
        else:
            lookback = 80  # Medium volatility → Default window

        # Ensure we have enough data for regression
        if i - lookback < 0:
            continue

        rolling_df = data.iloc[i - lookback:i]

        # Define independent (X_t-1) and dependent (ΔX) variables
        X_rolling = rolling_df['Close'].shift(1).dropna()
        y_rolling = rolling_df['Close'].diff().dropna()
        X_rolling = sm.add_constant(X_rolling)

        # Fit regression model
        model_rolling = sm.OLS(y_rolling, X_rolling).fit()
        alpha_hat, beta_hat = model_rolling.params
        t_beta_hat = model_rolling.tvalues[1]

        # Predict next day's price movement
        X_t = data.iloc[i]['Close']
        expected_change = alpha_hat + beta_hat * X_t

        # **Momentum Filter: Avoid trading if price is in a strong trend**
        SMA_10 = data.iloc[i]['SMA_10']
        if expected_change > 0 and X_t > SMA_10:  # Avoid buying when price is above SMA (trend still strong)
            position = 0
        elif expected_change < 0 and X_t < SMA_10:  # Avoid selling when price is below SMA (downtrend still strong)
            position = 0
        else:
            position = 1 if expected_change > 0 else -1  # Trade normally if no strong trend

        positions.append(position)

        # Compute return based on position
        next_day_return = (data.iloc[i + 1]['Close'] - X_t) / X_t
        strategy_return = position * next_day_return
        returns.append(strategy_return)

        # Update capital
        capital *= (1 + strategy_return)
        capital_series.append(capital)

    # Convert Results into a DataFrame
    performance_momentum = pd.DataFrame({
        'Date': data.index[100:len(data) - 1],
        'Capital': capital_series
    }).set_index('Date')

    # Compute Performance Metrics
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = (performance_momentum['Capital'].min() / performance_momentum['Capital'].max()) - 1
    cumulative_return = performance_momentum['Capital'].iloc[-1] 

    metrics_momentum = pd.DataFrame({
        "Strategy": ["Momentum-Filtered Mean-Reversion"],
        "Currency Pair": [currency_pair],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown": [max_drawdown],
        "Cumulative Return ": [cumulative_return]
    })

    # Plot Cumulative Performance
    plt.figure(figsize=(12, 6))
    plt.plot(performance_momentum['Capital'], label=f"{currency_pair} Momentum-Filtered Strategy", color="blue")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Cumulative Performance of Mean-Reversion Strategy with Momentum Filter ({currency_pair})")
    plt.xlabel("Date")
    plt.ylabel("Capital (Starting at 1 USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print Performance Metrics
    print("\nStrategy Performance Metrics:")
    print(metrics_momentum.to_string(index=False))

    return performance_momentum, metrics_momentum

# Run the strategy with user-defined dates
momentum_performance, momentum_metrics = mean_reversion_momentum_filter(
    ticker="EURUSD=X",
    start_date="2010-01-01",
    end_date="2024-12-31",
    currency_pair="EUR/USD"
)


#Refinement 03: mean_reversion_sharpe_filter

def mean_reversion_sharpe_filter(ticker, start_date, end_date, currency_pair=None):
    """
    Implements the Mean-Reversion Strategy with Momentum, Volatility Filter, and Sharpe Ratio Optimization.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol (e.g., "EURUSD=X").
        start_date (str): Start date for historical data download (format: "YYYY-MM-DD").
        end_date (str): End date for historical data download (format: "YYYY-MM-DD").
        currency_pair (str): The name of the forex pair (default: "EUR/USD").
    
    Returns:
        performance_sharpe (pd.DataFrame): DataFrame with cumulative performance results.
        metrics_sharpe (pd.DataFrame): DataFrame with strategy performance metrics.
    """

    # Step 1: Download Forex Data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if isinstance(data.columns, pd.MultiIndex):  # Check if the dataframe is a multiIndex
        data.columns = data.columns.get_level_values(0)

    # Data cleaning
    data = data[['Close']].dropna()

    # Compute rolling volatility (Standard deviation of returns over 30 days)
    data['Volatility'] = data['Close'].pct_change().rolling(window=30).std()

    # Compute 10-day simple moving average (SMA) for momentum filtering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()

    # Define a dynamic volatility threshold (e.g., 75th percentile of past volatility)
    volatility_threshold = data['Volatility'].quantile(0.75)

    # Initialize capital and performance tracking
    capital = 1  # Start with 1 USD
    capital_series = []
    returns = []
    positions = []

    # Implement the Mean-Reversion Strategy with Sharpe Optimization
    for i in range(100, len(data) - 1):  # Start from 100 to allow for lookback window
        # Define hybrid lookback window based on volatility
        volatility = data['Volatility'].iloc[i]
        if volatility > data['Volatility'].quantile(0.75):  
            lookback = 70  # High volatility → Shorter window
        elif volatility < data['Volatility'].quantile(0.25):  
            lookback = 100  # Low volatility → Longer window
        else:
            lookback = 80  # Medium volatility → Default window

        # Ensure we have enough data for regression
        if i - lookback < 0:
            continue

        # Volatility Filter: Skip trading if volatility is too high
        if volatility > volatility_threshold:
            capital_series.append(capital)
            positions.append(0)  # No trade
            continue

        rolling_df = data.iloc[i - lookback:i]

        # Define independent (X_t-1) and dependent (ΔX) variables
        X_rolling = rolling_df['Close'].shift(1).dropna()
        y_rolling = rolling_df['Close'].diff().dropna()
        X_rolling = sm.add_constant(X_rolling)

        # Fit regression model
        model_rolling = sm.OLS(y_rolling, X_rolling).fit()
        alpha_hat, beta_hat = model_rolling.params
        t_beta_hat = model_rolling.tvalues[1]

        # Predict next day's price movement
        X_t = data.iloc[i]['Close']
        expected_change = alpha_hat + beta_hat * X_t

        # **Momentum Filter: Avoid trading if price is in a strong trend**
        SMA_10 = data.iloc[i]['SMA_10']
        if expected_change > 0 and X_t > SMA_10:  # Avoid buying when price is above SMA (trend still strong)
            position = 0
        elif expected_change < 0 and X_t < SMA_10:  # Avoid selling when price is below SMA (downtrend still strong)
            position = 0
        else:
            position = 1 if expected_change > 0 else -1  # Trade normally if no strong trend

        # **Sharpe Ratio Optimization: Trade only if Expected Sharpe > 0.5**
        if len(returns) > 30:  # Ensure enough data for rolling Sharpe Ratio calculation
            rolling_sharpe = np.mean(returns[-30:]) / np.std(returns[-30:]) * np.sqrt(252) if np.std(returns[-30:]) > 0 else 0
            if rolling_sharpe < 0.5:
                position = 0  # Skip trade if Sharpe Ratio is too low

        positions.append(position)

        # Compute return based on position
        next_day_return = (data.iloc[i + 1]['Close'] - X_t) / X_t
        strategy_return = position * next_day_return
        returns.append(strategy_return)

        # Update capital
        capital *= (1 + strategy_return)
        capital_series.append(capital)

    # Convert Results into a DataFrame
    performance_sharpe = pd.DataFrame({
        'Date': data.index[100:len(data) - 1],
        'Capital': capital_series
    }).set_index('Date')

    # Compute Performance Metrics
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = (performance_sharpe['Capital'].min() / performance_sharpe['Capital'].max()) - 1
    cumulative_return = performance_sharpe['Capital'].iloc[-1]

    metrics_sharpe = pd.DataFrame({
        "Strategy": ["Sharpe-Optimized Mean-Reversion"],
        "Currency Pair": [currency_pair],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown": [max_drawdown],
        "Cumulative Return": [cumulative_return]
    })

    # Plot Cumulative Performance
    plt.figure(figsize=(12, 6))
    plt.plot(performance_sharpe['Capital'], label=f"{currency_pair} Sharpe-Optimized Strategy", color="blue")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Cumulative Performance of Mean-Reversion Strategy with Sharpe Optimization ({currency_pair})")
    plt.xlabel("Date")
    plt.ylabel("Capital (Starting at 1 USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print Performance Metrics
    print("\nStrategy Performance Metrics:")
    print(metrics_sharpe.to_string(index=False))

    return performance_sharpe, metrics_sharpe

sharpe_performance, sharpe_metrics = mean_reversion_sharpe_filter(
    ticker="EURUSD=X",
    start_date="2010-01-01",
    end_date="2024-12-31",
    currency_pair="EUR/USD"
)

#Refinement 04: hurst_regime_trading_strategy

# Function to compute the Hurst Exponent
def compute_hurst_exponent(ts, max_lag=20):
    """Compute the Hurst Exponent for a given time series."""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    hurst, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return hurst

# Fully integrated strategy with data download and Hurst-based trading logic
def hurst_regime_trading_strategy(ticker, start_date, end_date, stop_loss_threshold=None, currency_pair=None):
    """
    Implements the Hurst-Based Regime-Switching Strategy for a given forex pair.

    Parameters:
        ticker (str): The Yahoo Finance ticker symbol (e.g., "EURUSD=X").
        start_date (str): Start date for data download (format: "YYYY-MM-DD").
        end_date (str): End date for data download (format: "YYYY-MM-DD").
        stop_loss_threshold (float, optional): If provided, applies a stop-loss mechanism.
        currency_pair (str): The name of the forex pair (default: "EUR/USD").
    
    Returns:
        performance_df (pd.DataFrame): Cumulative performance results.
        metrics_df (pd.DataFrame): Strategy performance metrics.
    """

    # Step 1: Download Forex Data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    if isinstance(df.columns, pd.MultiIndex):  
        df.columns = df.columns.get_level_values(0)

    df.drop(columns=['Adj Close', 'Volume'], errors='ignore', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Step 2: Compute the Hurst Exponent and detect Market Regimes
    window = 100  
    hurst_values = []
    for i in range(len(df)):
        if i < window:
            hurst_values.append(np.nan)
        else:
            hurst_values.append(compute_hurst_exponent(df['Close'].iloc[i-window:i].values))

    df['Hurst'] = hurst_values
    df['Regime'] = np.where(df['Hurst'] < 0.5, 'Mean-Reverting', 'Trending')

    # Step 3: Initialize Trading Variables
    capital = 1  
    capital_series = []
    returns = []
    positions = []

    # Step 4: Implement Trading Logic
    for i in range(100, len(df) - 1):  
        hurst_value = df.iloc[i]['Hurst']
        
        if np.isnan(hurst_value) or 0.45 <= hurst_value <= 0.55:
            capital_series.append(capital)
            positions.append(0)
            continue

        X_t = df.iloc[i]['Close']
        X_t_1 = df.iloc[i - 1]['Close']

        #Mean-Reverting vs. Trending Logic
        if hurst_value < 0.5:  
            position = 1 if X_t < X_t_1 else -1  
        else:  
            position = 1 if X_t > X_t_1 else -1  

        positions.append(position)

        #Compute Return and Apply Stop-Loss if Enabled
        next_day_price = df.iloc[i + 1]['Close']
        next_day_return = (next_day_price - X_t) / X_t
        strategy_return = position * next_day_return

        if stop_loss_threshold and strategy_return < -stop_loss_threshold:
            strategy_return = -stop_loss_threshold  

        returns.append(strategy_return)

        # Update Capital
        capital *= (1 + strategy_return)
        capital_series.append(capital)

    #Step 5: Convert Results into a DataFrame
    performance_df = pd.DataFrame({
        'Date': df.index[100:len(df) - 1],
        'Capital': capital_series
    }).set_index('Date')

    #Step 6: Compute Performance Metrics
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = (performance_df['Capital'].min() / performance_df['Capital'].max()) - 1
    cumulative_return = performance_df['Capital'].iloc[-1]

    metrics_df = pd.DataFrame({
        "Strategy": ["Hurst Regime-Switching" + (" with Stop-Loss" if stop_loss_threshold else "")],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown": [max_drawdown],
        "Cumulative Return": [cumulative_return]
    })

    # Step 7: Plot Cumulative Performance
    plt.figure(figsize=(12, 6))
    plt.plot(performance_df['Capital'], label=f"Hurst Regime Strategy ({'Stop-Loss' if stop_loss_threshold else 'Basic'})", color="blue")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Cumulative Performance of Hurst Regime Strategy ({'Stop-Loss' if stop_loss_threshold else 'Basic'}) of ({currency_pair})")
    plt.xlabel("Date")
    plt.ylabel("Capital (Starting at 1 USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Step 8: Print Performance Metrics
    print("\nStrategy Performance Metrics:")
    print(metrics_df.to_string(index=False))

    return performance_df, metrics_df

#Run basic strategy
performance_hurst, metrics_hurst = hurst_regime_trading_strategy("EURUSD=X", start_date="2010-01-01", end_date="2024-12-31",currency_pair="EUR/USD")

#Run with stop-loss
performance_hurst_stoploss, metrics_hurst_stoploss = hurst_regime_trading_strategy("EURUSD=X", start_date="2010-01-01", end_date="2024-12-31", stop_loss_threshold=0.02,currency_pair="EUR/USD")


#Refinement 05: hurst_regime_trading_with_stoploss

def hybrid_trading_strategy(ticker, start_date, end_date, short_ma=20, long_ma=50, hurst_window=100, stop_loss_multiplier=1.5, take_profit_multiplier=2, currency_pair=None):
    """
    Implements the Hybrid Trend-Following & Mean-Reversion Strategy.

    Parameters:
        ticker (str): The Yahoo Finance ticker symbol (e.g., "EURUSD=X").
        start_date (str): Start date for data download (format: "YYYY-MM-DD").
        end_date (str): End date for data download (format: "YYYY-MM-DD").
        short_ma (int): Window for short-term moving average.
        long_ma (int): Window for long-term moving average.
        hurst_window (int): Lookback window for computing the Hurst exponent.
        stop_loss_multiplier (float): Multiplier for stop-loss calculation.
        take_profit_multiplier (float): Multiplier for take-profit calculation.
        currency_pair (str): The name of the forex pair (default: "EUR/USD").
    Returns:
        performance_df (pd.DataFrame): Cumulative performance results.
        metrics_df (pd.DataFrame): Strategy performance metrics.
    """

    # Download Forex Data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    if isinstance(df.columns, pd.MultiIndex):  
        df.columns = df.columns.get_level_values(0)

    df.drop(columns=['Adj Close', 'Volume'], errors='ignore', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Compute Moving Averages
    df['Short_MA'] = df['Close'].rolling(window=short_ma).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_ma).mean()

    # Compute Hurst Exponent
    hurst_values = [np.nan] * len(df)
    for i in range(hurst_window, len(df)):
        hurst_values[i] = compute_hurst_exponent(df['Close'].iloc[i-hurst_window:i].values)
    df['Hurst'] = hurst_values

    # Classify Market Regimes
    df['Market_Regime'] = 'Uncertain'
    df.loc[(df['Hurst'] > 0.55) & (df['Short_MA'] > df['Long_MA']), 'Market_Regime'] = 'Trending Up'
    df.loc[(df['Hurst'] > 0.55) & (df['Short_MA'] < df['Long_MA']), 'Market_Regime'] = 'Trending Down'
    df.loc[df['Hurst'] < 0.45, 'Market_Regime'] = 'Mean-Reverting'

    # Initialize Trading Variables
    capital = 1  
    capital_series = []
    returns = []
    positions = []

    # Implement Trading Logic
    for i in range(100, len(df) - 1):  
        price = df.iloc[i]['Close']
        mean_price = df.iloc[i]['Short_MA']
        long_ma = df.iloc[i]['Long_MA']
        market_regime = df.iloc[i]['Market_Regime']

        # Define Trading Logic Based on Regime
        if market_regime == 'Trending Up':
            if price > long_ma:
                position = 1
                stop_loss = price - stop_loss_multiplier * (price - long_ma)
                take_profit = price + take_profit_multiplier * (price - long_ma)
            else:
                position = 0

        elif market_regime == 'Trending Down':
            if price < long_ma:
                position = -1
                stop_loss = price + stop_loss_multiplier * (long_ma - price)
                take_profit = price - take_profit_multiplier * (long_ma - price)
            else:
                position = 0

        elif market_regime == 'Mean-Reverting':
            if price < mean_price:
                position = 1  
                stop_loss = price - stop_loss_multiplier * (mean_price - price)
                take_profit = mean_price  
            elif price > mean_price:
                position = -1  
                stop_loss = price + stop_loss_multiplier * (price - mean_price)
                take_profit = mean_price  
            else:
                position = 0

        else:
            position = 0  

        positions.append(position)

        # Compute Return and Apply Stop-Loss/Take-Profit
        next_day_price = df.iloc[i + 1]['Close']
        next_day_return = (next_day_price - price) / price
        strategy_return = position * next_day_return

        if position != 0:
            if (position == 1 and next_day_price <= stop_loss) or (position == -1 and next_day_price >= stop_loss):
                strategy_return = -stop_loss_multiplier * abs(price - mean_price) / price  
            elif (position == 1 and next_day_price >= take_profit) or (position == -1 and next_day_price <= take_profit):
                strategy_return = take_profit_multiplier * abs(price - mean_price) / price  

        returns.append(strategy_return)

        # Update Capital
        capital *= (1 + strategy_return)
        capital_series.append(capital)

    # Convert Results into a DataFrame
    performance_df = pd.DataFrame({
        'Date': df.index[100:len(df) - 1],
        'Capital': capital_series
    }).set_index('Date')

    # Compute Performance Metrics
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = (performance_df['Capital'].min() / performance_df['Capital'].max()) - 1
    cumulative_return = performance_df['Capital'].iloc[-1] 

    metrics_df = pd.DataFrame({
        "Strategy": ["Hybrid Trend-Following & Mean-Reversion"],
        "Sharpe Ratio": [sharpe_ratio],
        "Max Drawdown": [max_drawdown],
        "Cumulative Return ": [cumulative_return]
    })
    
    if currency_pair is None:
        currency_pair = ticker.replace("=X", "")

    # Plot Cumulative Performance
    # Plot Cumulative Performance
    plt.figure(figsize=(12, 6))
    plt.plot(performance_df['Capital'], label=f"{currency_pair} Hybrid Strategy", color="blue")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Cumulative Performance of Hybrid Trend-Following & Mean-Reversion Strategy ({currency_pair})")
    plt.xlabel("Date")
    plt.ylabel("Capital (Starting at 1 USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print Performance Metrics
    print("\nStrategy Performance Metrics:")
    print(metrics_df.to_string(index=False))

    return performance_df, metrics_df

# Usage
performance_hybrid, metrics_hybrid = hybrid_trading_strategy("EURUSD=X", "2010-01-01", "2024-12-31",currency_pair="EUR/USD")


#Q4)
# Subpart:  Check if the refined strategy works for any other currency pairs  
performance_hybrid, metrics_hybrid = hybrid_trading_strategy("EURGBP=X", "2010-01-01", "2024-12-31",currency_pair="EUR/GBP")
performance_hybrid, metrics_hybrid = hybrid_trading_strategy("GBPUSD=X", "2010-01-01", "2024-12-31",currency_pair="GBP/USD")
performance_hybrid, metrics_hybrid = hybrid_trading_strategy("JPYUSD=X", "2010-01-01", "2024-12-31",currency_pair="JPY/USD")


# Subpart:  Check if the refined strategy works for any other currency pairs  

def check_mean_reversion(ticker):
    """
    Check if mean reversion is present for a given currency pair.
    
    Parameters:
    - ticker : String representing the currency pair ticker (e.g., "EURUSD=X").
    
    Returns:
    - String : A message indicating whether mean reversion is present or not, including the ticker name.
    """
    # Download data
    data = yf.download(ticker, start='2010-01-01', end='2024-12-31', interval='1d')
    data = data[['Close']].dropna()
    
    # Create necessary columns
    data['Delta_Close'] = data['Close'].diff()
    data['Lagged_Close'] = data['Close'].shift(1)
    data = data.dropna()
    
    # Perform OLS regression
    X = sm.add_constant(data['Lagged_Close'])
    y = data['Delta_Close']
    model = sm.OLS(y, X).fit()
    
    # Extract coefficients and t-statistic
    alpha = model.params['const']
    beta = model.params['Lagged_Close']
    t_stat_beta = model.tvalues['Lagged_Close']
    
    # Check for mean reversion
    if alpha > 0 and beta < 0 and t_stat_beta < -1:  # 5% significance level
        return f"{ticker}: Mean reversion is present."
    else:
        return f"{ticker}: No mean reversion detected."

check_mean_reversion("EURUSD=X")
check_mean_reversion("EURGBP=X")
check_mean_reversion("GBPUSD=X")
check_mean_reversion("JPYUSD=X")
check_mean_reversion("GBPJPY=X")
check_mean_reversion("AUDNZD=X")
check_mean_reversion("EURAUD=X")
check_mean_reversion("EURCAD=X")

def calculate_correlation(pairs, start_date='2010-01-01', end_date='2024-12-31'):
    """
    Calculate the correlation matrix for selected currency pairs.
    """
    closes = pd.DataFrame()
    
    for pair in pairs:
        data = yf.download(pair, start=start_date, end=end_date, interval='1d')
        closes[pair] = data['Close']
    
    closes = closes.dropna()
    return closes.corr()

def get_least_correlated_pairs(pairs, start_date='2010-01-01', end_date='2024-12-31'):
    """
    Selects the three least correlated currency pairs.
    """
    corr_matrix = calculate_correlation(pairs, start_date, end_date)
    
    # Generate all possible sets of 3 pairs
    pair_combinations = list(combinations(pairs, 3))
    
    # Calculate the average correlation for each combination
    min_avg_corr = float('inf')
    best_combination = None
    
    for comb in pair_combinations:
        avg_corr = (corr_matrix.loc[comb, comb].sum().sum() - len(comb)) / (len(comb) * (len(comb) - 1))
        if avg_corr < min_avg_corr:
            min_avg_corr = avg_corr
            best_combination = comb
    
    return best_combination

candidate_pairs = ["EURUSD=X", "EURGBP=X", "EURAUD=X", "GBPUSD=X", "AUDNZD=X"]
least_correlated_pairs = get_least_correlated_pairs(candidate_pairs)
print("Least correlated pairs:", least_correlated_pairs)

def check_cointegration_johansen(pairs, start_date='2010-01-01', end_date='2024-12-31'):
    """
    Checks if the selected pairs are cointegrated using the Johansen test.
    """
    data = yf.download(pairs, start=start_date, end=end_date, interval='1d')['Close'].dropna()
    
    # Perform Johansen cointegration test
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    
    # Check if trace statistic is above the critical value (5% level)
    trace_stat = result.lr1
    critical_values = result.cvt[:, 1]  # 5% critical values
    cointegrated = np.any(trace_stat > critical_values)
    
    if cointegrated:
        print("The selected pairs are cointegrated based on the Johansen test.")
    else:
        print("The selected pairs are not cointegrated based on the Johansen test.")

check_cointegration_johansen(least_correlated_pairs)


# Function to implement mean reversion strategy for multiple pairs

def jointly_trading_strategy_v1(pairs, start_date='2010-01-01', end_date='2024-12-31', estimation_window=80):
    """
    Implements a simultaneous trading strategy for multiple currency pairs using a rolling regression approach.
    
    Parameters:
    - pairs: List of currency pairs (e.g., ["EURUSD=X", "EURGBP=X", "EURAUD=X"])
    - start_date: Start date for historical data
    - end_date: End date for historical data
    - estimation_window: Rolling regression window size
    
    Returns:
    - DataFrame with trading signals and portfolio performance
    """
    
    # Download historical price data
    data = yf.download(pairs, start=start_date, end=end_date, interval='1d')['Close']
    data = data.dropna()  # Remove missing values
    
    # Initialize variables
    signals = pd.DataFrame(index=data.index, columns=pairs)
    portfolio_value = [1.0]  # Starting capital of 1 unit
    positions = {pair: 0 for pair in pairs}  # Track positions for each pair
    capital = 1.0  # Initial capital
    
    # Iterate over the data with a rolling window
    for i in range(estimation_window, len(data)):
        current_data = data.iloc[i - estimation_window:i]
        signals_at_t = {}
        
        # Estimate mean-reversion parameters for each pair
        for pair in pairs:
            X = current_data[pair].shift(1).iloc[1:].values.reshape(-1, 1)
            X = sm.add_constant(X)  # Add intercept
            y = current_data[pair].diff().iloc[1:].values
            
            # Manually implement rolling regression
            try:
                model = sm.OLS(y, X)
                results = model.fit()
                
                # Extract parameters
                alpha = results.params[0]  # Intercept
                beta = results.params[1]  # Coefficient
                t_stat = results.tvalues[1]  # t-statistic for beta
                
                # Generate trading signal
                if alpha > 0 and beta < 0 and t_stat < -1:
                    expected_change = alpha + beta * current_data[pair].iloc[-1]
                    if expected_change > 0:
                        signals_at_t[pair] = 1  # (long)
                    else:
                        signals_at_t[pair] = -1  #(short)
                else:
                    signals_at_t[pair] = 0  # No position
            except:
                signals_at_t[pair] = 0  # Skip if regression fails
        
        # Check if all pairs have the same signal direction
        if all(signal == 1 for signal in signals_at_t.values()):
            # Go long all pairs
            for pair in pairs:
                positions[pair] = 1
        elif all(signal == -1 for signal in signals_at_t.values()):
            # Go short all pairs
            for pair in pairs:
                positions[pair] = -1
        else:
            # Close all positions
            for pair in pairs:
                positions[pair] = 0
        
        # Update portfolio value
        daily_return = 0
        for pair in pairs:
            if positions[pair] == 1:
                daily_return += data[pair].iloc[i] / data[pair].iloc[i - 1] - 1
            elif positions[pair] == -1:
                daily_return += 1 - data[pair].iloc[i] / data[pair].iloc[i - 1]
        
        capital *= (1 + daily_return / len(pairs))  # Equal-weighted portfolio
        portfolio_value.append(capital)
    
    # Calculate portfolio returns
    portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    cumulative_returns = (pd.Series(portfolio_value) - 1)*100
    
    # Calculate Sharpe Ratio
    risk_free_rate = 0  # Assume risk-free rate is 0
    sharpe_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate Max Drawdown
    cumulative_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
        
    # Plot cumulative returns over time
    pairs_str = ", ".join(pairs)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[estimation_window:], cumulative_returns[1:], label='Cumulative Returns (%)', color='blue')
    plt.fill_between(data.index[estimation_window:], cumulative_returns[1:], alpha=0.1, color='blue')
    plt.title(f'Portfolio Performance Over Time for Pairs: {pairs_str}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print performance metrics
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    return portfolio_value, sharpe_ratio, max_drawdown

# Example usage
pairs = ["EURUSD=X", "EURGBP=X", "EURAUD=X"]
pairs_1= ["EURGBP=X", "EURAUD=X", "GBPUSD=X"]
portfolio_value, sharpe_ratio, max_drawdown = jointly_trading_strategy_v1(pairs)