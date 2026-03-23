import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO


# Backtesting
def run_backtest(rl_model, prices, predicted_prices, sentiments):

    print("Running backtest simulation...")

    initial_capital = 10000.0
    capital         = initial_capital
    holding         = False
    buy_price       = 0.0
    btc_held        = 0.0
    total_profit    = 0.0
    steps_holding   = 0
    max_hold_days   = 30   # must match rl_agent.py

    history = []

    for i in range(len(prices)):
        current_price   = prices[i]
        predicted_price = predicted_prices[i]
        sentiment       = sentiments[i]

        # Calculate real momentum
        if i > 0:
            momentum = (current_price - prices[i - 1]) / prices[i - 1]
        else:
            momentum = 0.0

        # Normalised steps holding
        norm_steps = steps_holding / max_hold_days

        # Build observation matching rl_agent.py exactly
        obs = np.array([
            current_price   / 100000,
            predicted_price / 100000,
            sentiment,
            float(holding),
            norm_steps,      # ← real value not 0.0
            momentum * 100   # ← real value not 0.0
        ], dtype=np.float32)

        # Force sell if held too long
        if holding and steps_holding >= max_hold_days:
            action       = 2
            action_label = "SELL"
        else:
            action, _    = rl_model.predict(obs, deterministic=True)
            action       = int(action)
            action_map   = {0: "BUY", 1: "HOLD", 2: "SELL"}
            action_label = action_map[action]

        # Execute action
        # Buy
        if action == 0:
            if not holding:
                btc_held      = capital / current_price
                buy_price     = current_price
                holding       = True
                steps_holding = 0

        # Hold
        elif action == 1:
            if holding:
                steps_holding += 1

        # Sell
        elif action == 2:
            if holding:
                capital       = btc_held * current_price
                profit        = (current_price - buy_price) / buy_price
                total_profit += profit
                holding       = False
                btc_held      = 0.0
                steps_holding = 0

        # Portfolio value
        if holding:
            portfolio_value = btc_held * current_price
        else:
            portfolio_value = capital

        history.append({
            "day":             i,
            "price":           current_price,
            "predicted_price": predicted_price,
            "action":          action_label,
            "holding":         holding,
            "steps_holding":   steps_holding,
            "portfolio_value": portfolio_value,
            "capital":         capital
        })

    df = pd.DataFrame(history)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/backtest_history.csv", index=False)
    print(f" Backtest complete. {len(df)} trading days simulated.")
    return df


# Cumulative Return
def calculate_cumulative_return(history_df, initial_capital=10000.0):

    final_value      = history_df["portfolio_value"].iloc[-1]
    cumulative_return = (final_value - initial_capital) / initial_capital * 100

    print(f"\n Cumulative Return:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value:     ${final_value:,.2f}")
    print(f"   Total Return:    {cumulative_return:+.2f}%")

    return cumulative_return


# Sharpe Ratio
def calculate_sharpe_ratio(history_df, risk_free_rate=0.02):

    # Calculate daily returns
    portfolio_values  = history_df["portfolio_value"].values
    daily_returns     = np.diff(portfolio_values) / portfolio_values[:-1]

    # Annualise the risk free rate to daily
    daily_risk_free   = risk_free_rate / 252  # 252 trading days per year

    # Excess returns above risk free rate
    excess_returns    = daily_returns - daily_risk_free

    # Sharpe ratio (annualised)
    if np.std(excess_returns) == 0:
        sharpe = 0.0
    else:
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe = sharpe * np.sqrt(252)  # annualise

    print(f"\n Sharpe Ratio:")
    print(f"   Value: {sharpe:.4f}")
    if sharpe > 2:
        print(f"   Rating: Excellent ✅")
    elif sharpe > 1:
        print(f"   Rating: Good ✅")
    elif sharpe > 0:
        print(f"   Rating: Acceptable ⚠️")
    else:
        print(f"   Rating: Poor ❌ (worse than risk-free investment)")

    return sharpe


# Maximum Drawdown 
def calculate_max_drawdown(history_df):

    portfolio_values = history_df["portfolio_value"].values

    # Track running peak
    peak         = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak * 100
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    print(f"\n Maximum Drawdown:")
    print(f"   Value: {max_drawdown:.2f}%")
    if max_drawdown > -10:
        print(f"   Rating: Low Risk ✅")
    elif max_drawdown > -25:
        print(f"   Rating: Moderate Risk ⚠️")
    else:
        print(f"   Rating: High Risk ❌")

    return max_drawdown


# Buy and Hold Benchmark
def calculate_buy_and_hold(prices, initial_capital=10000.0):

    btc_bought    = initial_capital / prices[0]
    final_value   = btc_bought * prices[-1]
    total_return  = (final_value - initial_capital) / initial_capital * 100

    print(f"\n Buy & Hold Benchmark:")
    print(f"   Buy price (day 1):  ${prices[0]:,.2f}")
    print(f"   Sell price (last):  ${prices[-1]:,.2f}")
    print(f"   Final Value:        ${final_value:,.2f}")
    print(f"   Total Return:       {total_return:+.2f}%")

    return total_return


# Plot Backtest Results 
def plot_backtest(history_df, prices, initial_capital=10000.0):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top chart: Bitcoin price with BUY/SELL markers 
    ax1.plot(history_df["price"], color="blue",
             linewidth=1.5, label="BTC Price")

    # Mark BUY points
    buy_days  = history_df[history_df["action"] == "BUY"]
    sell_days = history_df[history_df["action"] == "SELL"]

    ax1.scatter(buy_days["day"],  buy_days["price"],
                color="green", marker="^", s=100,
                label="BUY",  zorder=5)
    ax1.scatter(sell_days["day"], sell_days["price"],
                color="red",   marker="v", s=100,
                label="SELL", zorder=5)

    ax1.set_title("Bitcoin Price with RL Agent BUY/SELL Decisions")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom chart: Portfolio value vs buy and hold
    # Buy and hold line
    btc_bought    = initial_capital / prices[0]
    buy_hold_vals = [btc_bought * p for p in prices[:len(history_df)]]

    ax2.plot(history_df["portfolio_value"],
             color="orange", linewidth=2,
             label="RL Agent Portfolio")
    ax2.plot(buy_hold_vals,
             color="blue",   linewidth=2,
             linestyle="--", label="Buy & Hold")
    ax2.axhline(y=initial_capital, color="grey",
                linestyle=":",     label="Initial Capital")

    ax2.set_title("RL Agent Portfolio Value vs Buy & Hold Strategy")
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Portfolio Value (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/backtest_results.png")
    plt.show()
    print(" Backtest chart saved to data/backtest_results.png")


# Full Backtest Report 
def full_backtest_report(rl_model, prices,
                         predicted_prices, sentiments):
    """
    Runs the complete backtest and prints a full report.
    """
    print("\n" + "=" * 55)
    print("         RL AGENT BACKTEST REPORT")
    print("=" * 55)

    # Run simulation
    history_df = run_backtest(
        rl_model, prices, predicted_prices, sentiments
    )

    # Calculate all metrics
    cum_return   = calculate_cumulative_return(history_df)
    sharpe       = calculate_sharpe_ratio(history_df)
    max_dd       = calculate_max_drawdown(history_df)
    bh_return    = calculate_buy_and_hold(prices)

    # Count trades
    buys  = len(history_df[history_df["action"] == "BUY"])
    sells = len(history_df[history_df["action"] == "SELL"])
    holds = len(history_df[history_df["action"] == "HOLD"])

    print(f"\n Trading Activity:")
    print(f"   BUY  actions:  {buys}")
    print(f"   SELL actions:  {sells}")
    print(f"   HOLD actions:  {holds}")

    print(f"\n{'=' * 55}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Cumulative Return:  {cum_return:+.2f}%")
    print(f"  Sharpe Ratio:       {sharpe:.4f}")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Buy & Hold Return:  {bh_return:+.2f}%")
    print(f"  vs Buy & Hold:      "
          f"{'+' if cum_return > bh_return else ''}"
          f"{cum_return - bh_return:.2f}%")
    print(f"{'=' * 55}")

    # Plot results
    plot_backtest(history_df, prices)

    return {
        "cumulative_return": cum_return,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_dd,
        "buy_hold_return":   bh_return,
        "num_trades":        buys
    }


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ))

    from modules.data_collector  import get_price_data, get_news_headlines
    from modules.lstm_model      import prepare_data, predict_next_price
    from modules.sentiment       import (load_sentiment_model,
                                         analyse_headlines,
                                         get_overall_sentiment)
    from modules.rl_agent        import prepare_rl_data
    from tensorflow.keras.models import load_model as tf_load_model

    # Load data 
    print(" Loading data...")
    df = get_price_data()
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df["Close"] = df["Close"].astype(float)
    df = df.reset_index(drop=True)

    # LSTM predictions 
    print(" Generating LSTM predictions...")
    lstm_model         = tf_load_model("models/lstm_model.keras")
    _, _, _, _, scaler = prepare_data(df)

    all_scaled    = scaler.transform(df[["Close"]].values)
    predicted_all = []
    for i in range(60, len(all_scaled)):
        seq  = all_scaled[i - 60:i]
        seq  = np.expand_dims(seq, axis=0)
        pred = lstm_model.predict(seq, verbose=0)
        predicted_all.append(
            float(scaler.inverse_transform(pred)[0][0])
        )

    # Sentiment 
    print(" Getting sentiment...")
    headlines       = get_news_headlines()
    sentiment_model = load_sentiment_model()
    from modules.sentiment import analyse_headlines, get_overall_sentiment
    sentiment_df    = analyse_headlines(headlines, sentiment_model)
    overall_sent    = get_overall_sentiment(sentiment_df)

    # Prepare arrays 
    prices, predicted_prices, sentiments = prepare_rl_data(
        df, np.array(predicted_all), overall_sent
    )

    # Load RL agent
    print(" Loading RL agent...")
    rl_model = PPO.load("models/rl_agent")

    # Run full backtest 
    results = full_backtest_report(
        rl_model, prices, predicted_prices, sentiments
    )