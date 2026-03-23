import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os


# Trading Environment
class CryptoTradingEnv(gym.Env):

    def __init__(self, prices, predicted_prices, sentiments,
                 max_hold_days=30):
        super().__init__()

        self.prices           = prices
        self.predicted_prices = predicted_prices
        self.sentiments       = sentiments
        self.max_steps        = len(prices) - 1
        self.max_hold_days    = max_hold_days  # force sell after this many days

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (6,),  
            dtype = np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self.holding       = False
        self.buy_price     = 0.0
        self.total_profit  = 0.0
        self.steps_holding = 0
        return self._get_observation(), {}

    def _get_observation(self):
        current_price   = self.prices[self.current_step]
        predicted_price = self.predicted_prices[self.current_step]
        sentiment       = self.sentiments[self.current_step]

        # Price momentum 
        if self.current_step > 0:
            momentum = (current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        else:
            momentum = 0.0

        return np.array([
            current_price   / 100000,
            predicted_price / 100000,
            sentiment,
            float(self.holding),
            self.steps_holding / self.max_hold_days,  # normalised hold duration
            momentum * 100   # price momentum
        ], dtype=np.float32)

    def step(self, action):
        current_price   = self.prices[self.current_step]
        predicted_price = self.predicted_prices[self.current_step]
        sentiment       = self.sentiments[self.current_step]
        reward          = 0.0

        # Price momentum
        if self.current_step > 0:
            momentum = (current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        else:
            momentum = 0.0

        # Force sell if held too long
        if self.holding and self.steps_holding >= self.max_hold_days:
            action = 2

        # Buy
        if action == 0:
            if not self.holding:
                self.holding       = True
                self.buy_price     = current_price
                self.steps_holding = 0

                # Only reward buying when conditions are good
                predicted_up  = predicted_price > current_price
                sentiment_ok  = sentiment >= -0.3
                momentum_ok   = momentum > -0.01

                if predicted_up and sentiment_ok:
                    reward = 1.0   # strong reward for smart buy
                elif predicted_up or sentiment_ok:
                    reward = 0.3   # partial reward
                else:
                    reward = -1.0  # punish buying into bad conditions
            else:
                reward = -1.0  # punish buying when already holding

        # Hold
        elif action == 1:
            if self.holding:
                self.steps_holding += 1
                current_profit = (current_price - self.buy_price) / self.buy_price

                # Reward holding a winning position
                if current_profit > 0.02:    # more than 2% profit
                    reward = current_profit * 3.0
                elif current_profit > 0:     # small profit, be patient
                    reward = 0.2
                elif current_profit > -0.05: # small loss, acceptable
                    reward = -0.1
                else:                        # large loss, should have sold
                    reward = current_profit * 2.0

            else:
                # NOT holding — reward waiting for right conditions
                predicted_up = predicted_price > current_price
                if not predicted_up:
                    reward = 0.3   # good decision to wait when price dropping
                else:
                    reward = 0.0   # neutral — could have bought

        # Sell
        elif action == 2:
            if self.holding:
                profit = (current_price - self.buy_price) / self.buy_price

                if profit > 0.05:    # more than 5% profit — great sell
                    reward = profit * 20
                elif profit > 0:     # small profit — ok sell
                    reward = profit * 10
                elif profit > -0.05: # small loss — acceptable
                    reward = profit * 5
                else:                # large loss — bad trade
                    reward = profit * 15

                self.total_profit  += profit
                self.holding        = False
                self.buy_price      = 0.0
                self.steps_holding  = 0
            else:
                reward = -1.0  # punish selling when not holding

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, False, {}


# Preparing Data for RL
def prepare_rl_data(df, predicted_prices, overall_sentiment):
    prices = df["Close"].values.astype(float)

    min_len          = min(len(prices), len(predicted_prices))
    prices           = prices[-min_len:]
    predicted_prices = predicted_prices[-min_len:]
    sentiments       = np.full(min_len, overall_sentiment)

    print(f" RL data prepared. {min_len} trading days available.")
    return prices, predicted_prices, sentiments


# Training RL agent
def train_rl_agent(env):
    print("Training RL agent...")
    print("This will take 5-10 minutes...")

    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose       = 1,
        learning_rate = 0.0003,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 15,
        gamma         = 0.99,
        ent_coef      = 0.02   # higher exploration
    )

    model.learn(total_timesteps=100000)  

    os.makedirs("models", exist_ok=True)
    model.save("models/rl_agent")
    print(" RL agent saved to models/rl_agent.zip")
    return model


# Get trading decision
def get_rl_action(model, current_price, predicted_price,
                  sentiment, holding=False, total_profit=0.0):
    # Price momentum (0 at inference time since we only have current)
    momentum = 0.0

    obs = np.array([
        current_price   / 100000,
        predicted_price / 100000,
        sentiment,
        float(holding),
        0.0,       # steps_holding normalised
        momentum   # momentum
    ], dtype=np.float32)

    action, _    = model.predict(obs, deterministic=True)
    action_map   = {0: "BUY", 1: "HOLD", 2: "SELL"}
    action_label = action_map[int(action)]

    print(f" RL Decision: {action_label}")
    return int(action), action_label


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
    from tensorflow.keras.models import load_model as tf_load_model

    # Loading Data
    print(" Loading price data...")
    df = get_price_data()
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df["Close"] = df["Close"].astype(float)
    df = df.reset_index(drop=True)

    # LSTM prediction
    print(" Getting LSTM predictions...")
    lstm_model         = tf_load_model("models/lstm_model.keras")
    _, _, _, _, scaler = prepare_data(df)
    current_price      = float(df["Close"].iloc[-1])
    predicted_price    = predict_next_price(df, scaler, lstm_model)

    # Sentiment
    print(" Getting sentiment...")
    headlines       = get_news_headlines()
    sentiment_model = load_sentiment_model()
    sentiment_df    = analyse_headlines(headlines, sentiment_model)
    overall_sent    = get_overall_sentiment(sentiment_df)

    # Prepare RL data
    all_scaled    = scaler.transform(df[["Close"]].values)
    predicted_all = []
    for i in range(60, len(all_scaled)):
        seq  = all_scaled[i - 60:i]
        seq  = np.expand_dims(seq, axis=0)
        pred = lstm_model.predict(seq, verbose=0)
        predicted_all.append(
            float(scaler.inverse_transform(pred)[0][0])
        )

    prices, predicted_prices, sentiments = prepare_rl_data(
        df, np.array(predicted_all), overall_sent
    )

    # Train agent
    env      = CryptoTradingEnv(prices, predicted_prices, sentiments)
    rl_model = train_rl_agent(env)

    # Get trading decision
    action, label = get_rl_action(
        rl_model, current_price, predicted_price, overall_sent
    )

    print(f"\n Trading Decision:")
    print(f"   Current BTC Price:   ${current_price:,.2f}")
    print(f"   Predicted BTC Price: ${predicted_price:,.2f}")
    print(f"   Sentiment Score:     {overall_sent:.2f}")
    print(f"   Recommendation:      {label}")