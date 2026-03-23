# 🤖 Crypto Financial Advisor Bot

A cryptocurrency financial advisor bot that helps non-technical 
users make informed investment decisions using AI and machine learning.

## How to Run

1. Make sure Ollama is running with LLaMA 3.2:
```
ollama run llama3.2
```

2. Activate virtual environment:
```
venv\Scripts\activate
```

3. Run the app:
```
streamlit run app.py
```

4. Open browser at:
```
http://localhost:8501
```

## Project Structure
```
crypto_advisor_bot/
│
├── data/
│   ├── price_data.csv
│   ├── sentiment_results.csv
│   ├── backtest_history.csv
│   ├── lstm_predictions.png
│   ├── model_comparison.png
│   └── backtest_results.png
│
├── models/
│   ├── lstm_model.keras
│   ├── transformer_model.keras
│   └── rl_agent.zip
│
├── modules/
│   ├── __init__.py
│   ├── data_collector.py
│   ├── lstm_model.py
│   ├── sentiment.py
│   ├── rl_agent.py
│   ├── llm_rag.py
│   ├── transformer_model.py
│   └── rl_backtest.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Module Descriptions

| Module | Description |
|--------|-------------|
| data_collector.py | Downloads BTC prices and news headlines |
| lstm_model.py | LSTM price prediction model |
| transformer_model.py | Transformer price prediction model |
| sentiment.py | FinBERT sentiment analysis |
| rl_agent.py | PPO reinforcement learning agent |
| rl_backtest.py | Backtests RL agent performance |
| llm_rag.py | LLM + RAG conversational layer |
| app.py | Streamlit chat interface |

## Technologies Used

| Technology | Purpose |
|------------|---------|
| TensorFlow / Keras | LSTM and Transformer models |
| HuggingFace FinBERT | Sentiment analysis |
| Stable Baselines3 | Reinforcement learning (PPO) |
| Ollama / LLaMA 3.2 | LLM conversation layer |
| Streamlit | Chat interface |
| yfinance | Bitcoin price data |
| feedparser | CoinDesk news headlines |

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| LSTM | MAE | $3,035.88 |
| LSTM | RMSE | $3,899.90 |
| LSTM | Directional Accuracy | 50.1% |
| Transformer | MAE | $19,801.11 |
| Transformer | RMSE | $22,701.66 |
| Transformer | Directional Accuracy | 49.0% |
| RL Agent | Cumulative Return | +186.18% |
| RL Agent | Sharpe Ratio | 0.5669 |
| RL Agent | Max Drawdown | -35.71% |
| RL Agent | Buy & Hold Return | +83.08% |