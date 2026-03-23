import ollama

# System Prompt 
# This tells the LLM how to behave
SYSTEM_PROMPT = """
You are a friendly and helpful cryptocurrency financial advisor bot 
designed specifically for non-technical users who are new to crypto investing.

Your job is to explain investment insights in simple, clear, everyday language.
Avoid using technical jargon. If you must use a technical term, explain it simply.

You have access to the following real-time analysis from our system:
- LSTM price prediction (a machine learning model that predicts future prices)
- Sentiment analysis (analysis of current crypto news mood)
- Reinforcement learning recommendation (an AI that decides BUY, HOLD or SELL)

You ONLY answer questions related to:
- Bitcoin and cryptocurrency prices
- Market sentiment and news
- Investment recommendations (BUY/HOLD/SELL)
- The analysis provided in the context below

STRICT RULES you must ALWAYS follow:
1. ONLY use the provided analysis context for data
2. NEVER answer questions unrelated to cryptocurrency
3. NEVER make up prices or data not in the context
4. ALWAYS remind users crypto investing carries risk
5. ALWAYS keep explanations short, friendly and easy to understand
6. ALWAYS refer to the context data when answering price questions
7. If asked anything unrelated to crypto investing,
   respond with exactly:
   "I can only assist with cryptocurrency investment
   questions. Please ask me about Bitcoin prices,
   market sentiment, or investment recommendations."

You must REFUSE to answer questions about:
- Food, recipes, cooking
- Sports, entertainment, movies
- General knowledge or trivia
- Anything not related to crypto investing
"""

# Build Context 
def build_context(current_price, predicted_price,
                  sentiment_score, rl_action_label):

    # Calculate expected price change
    price_change     = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100

    # Convert sentiment score to human readable label
    if sentiment_score > 0.2:
        sentiment_label = "Positive  (people are optimistic about crypto)"
    elif sentiment_score < -0.2:
        sentiment_label = "Negative  (people are pessimistic about crypto)"
    else:
        sentiment_label = "Neutral   (mixed opinions about crypto)"

    # Convert price change to human readable label
    if price_change_pct > 0:
        trend = f"expected to RISE by ${abs(price_change):,.2f} ({abs(price_change_pct):.2f}%)"
    else:
        trend = f"expected to FALL by ${abs(price_change):,.2f} ({abs(price_change_pct):.2f}%)"

    context = f"""
=== CURRENT SYSTEM ANALYSIS ===

   Price Analysis:
   Current Bitcoin Price:   ${current_price:,.2f}
   Predicted Tomorrow:      ${predicted_price:,.2f}
   Price Trend:             {trend}

   News Sentiment Analysis:
   Sentiment Score:         {sentiment_score:.2f} (range: -1 to +1)
   Market Mood:             {sentiment_label}

   AI Trading Recommendation:
   Reinforcement Learning:  {rl_action_label}

================================
"""
    return context


# Ask the Advisor 
def ask_advisor(user_question, context, chat_history=[]):
    
    # Build the full message list
    messages = [
        {
            "role":    "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Add chat history so LLM remembers previous messages
    for msg in chat_history:
        messages.append(msg)

    # Add the current question with context
    messages.append({
        "role": "user",
        "content": f"""
Here is the latest analysis from our system:
{context}

User question: {user_question}
"""
    })

    # Send to Ollama LLaMA model
    response = ollama.chat(
        model    = "llama3.2",
        messages = messages
    )

    return response["message"]["content"]


# Format Response 
def format_response(response):

    return response.strip()


# Test
if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ))

    from modules.data_collector import get_price_data, get_news_headlines
    from modules.lstm_model     import prepare_data, predict_next_price
    from modules.sentiment      import (load_sentiment_model,
                                        analyse_headlines,
                                        get_overall_sentiment)
    from modules.rl_agent       import get_rl_action
    from tensorflow.keras.models import load_model as tf_load_model
    from stable_baselines3       import PPO

    # Load all module outputs 
    print("Loading all module outputs...")

    # Price data
    df = get_price_data()
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df["Close"] = df["Close"].astype(float)
    df = df.reset_index(drop=True)

    # LSTM prediction
    lstm_model      = tf_load_model("models/lstm_model.keras")
    _, _, _, _, scaler = prepare_data(df)
    current_price   = float(df["Close"].iloc[-1])
    predicted_price = predict_next_price(df, scaler, lstm_model)

    # Sentiment
    headlines       = get_news_headlines()
    sentiment_model = load_sentiment_model()
    sentiment_df    = analyse_headlines(headlines, sentiment_model)
    sentiment_score = get_overall_sentiment(sentiment_df)

    # RL decision
    rl_model        = PPO.load("models/rl_agent")
    action, label   = get_rl_action(
        rl_model, current_price, predicted_price, sentiment_score
    )

    # Build context
    context = build_context(
        current_price, predicted_price,
        sentiment_score, label
    )
    print("\n Context built successfully:")
    print(context)

    # Test questions
    print("\n Testing LLM responses...")
    print("=" * 60)

    test_questions = [
        "Should I buy Bitcoin today?",
        "Why is the market mood negative?",
        "What is the predicted price for tomorrow?"
    ]

    chat_history = []
    for question in test_questions:
        print(f"\n👤 User: {question}")
        response = ask_advisor(question, context, chat_history)
        response = format_response(response)
        print(f"🤖 Bot: {response}")
        print("-" * 60)

        # Add to chat history for multi turn conversation
        chat_history.append({
            "role": "user", "content": question
        })
        chat_history.append({
            "role": "assistant", "content": response
        })