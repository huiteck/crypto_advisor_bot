import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
from tensorflow.keras.models import load_model as tf_load_model
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.data_collector    import get_price_data, get_news_headlines
from modules.lstm_model        import prepare_data, predict_next_price
from modules.transformer_model import (prepare_data as transformer_prepare,
                                       predict_next_price as transformer_predict)
from modules.sentiment         import (load_sentiment_model,
                                       analyse_headlines,
                                       get_overall_sentiment)
from modules.rl_agent          import get_rl_action
from modules.rl_backtest       import (run_backtest,
                                       calculate_cumulative_return,
                                       calculate_sharpe_ratio,
                                       calculate_max_drawdown,
                                       calculate_buy_and_hold)
from modules.llm_rag           import build_context, ask_advisor, format_response

#  Page Configuration 
st.set_page_config(
    page_title = "Crypto Advisor Bot",
    page_icon  = "🤖",
    layout     = "wide"
)

# Load All Models
@st.cache_resource
def load_all_models():
    with st.spinner(" Loading all models... please wait..."):

        df = get_price_data()
        df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
        df["Close"] = df["Close"].astype(float)
        df = df.reset_index(drop=True)

        lstm_model         = tf_load_model("models/lstm_model.keras")
        _, _, _, _, scaler = prepare_data(df)

        transformer_model              = tf_load_model("models/transformer_model.keras")
        _, _, _, _, transformer_scaler = transformer_prepare(df)

        rl_model        = PPO.load("models/rl_agent")
        sentiment_model = load_sentiment_model()

    return (df, lstm_model, scaler,
            transformer_model, transformer_scaler,
            rl_model, sentiment_model)


# Get Live Analysis
@st.cache_data(ttl=300)
def get_live_analysis():
    (df, lstm_model, scaler,
     transformer_model, transformer_scaler,
     rl_model, sentiment_model) = load_all_models()

    current_price     = float(df["Close"].iloc[-1])
    lstm_price        = predict_next_price(df, scaler, lstm_model)
    transformer_price = transformer_predict(
        df, transformer_scaler, transformer_model
    )

    headlines       = get_news_headlines()
    sentiment_df    = analyse_headlines(headlines, sentiment_model)
    sentiment_score = get_overall_sentiment(sentiment_df)

    action, action_label = get_rl_action(
        rl_model, current_price, lstm_price, sentiment_score
    )

    # Generate LSTM predictions for backtest
    all_scaled    = scaler.transform(df[["Close"]].values)
    predicted_all = []
    for i in range(60, len(all_scaled)):
        seq  = all_scaled[i - 60:i]
        seq  = np.expand_dims(seq, axis=0)
        pred = lstm_model.predict(seq, verbose=0)
        predicted_all.append(
            float(scaler.inverse_transform(pred)[0][0])
        )

    prices        = df["Close"].values.astype(float)
    min_len       = min(len(prices), len(predicted_all))
    prices_bt     = prices[-min_len:]
    predicted_bt  = np.array(predicted_all[-min_len:])
    sentiments_bt = np.full(min_len, sentiment_score)

    history_df = run_backtest(
        rl_model, prices_bt, predicted_bt, sentiments_bt
    )
    cum_return = calculate_cumulative_return(history_df)
    sharpe     = calculate_sharpe_ratio(history_df)
    max_dd     = calculate_max_drawdown(history_df)
    bh_return  = calculate_buy_and_hold(prices_bt)

    return {
        "current_price":     current_price,
        "lstm_price":        lstm_price,
        "transformer_price": transformer_price,
        "sentiment_score":   sentiment_score,
        "action_label":      action_label,
        "cum_return":        cum_return,
        "sharpe":            sharpe,
        "max_dd":            max_dd,
        "bh_return":         bh_return,
        "history_df":        history_df
    }


# Load Everything
(df, lstm_model, scaler,
 transformer_model, transformer_scaler,
 rl_model, sentiment_model) = load_all_models()

analysis = get_live_analysis()

# Initialise Page State
if "page" not in st.session_state:
    st.session_state.page = "chat"  # default page is chat


# Sidebar
with st.sidebar:
    st.header(" Live Market Analysis")
    st.caption("Updates every 5 minutes")
    st.divider()

    # BTC Price Chart 
    st.subheader(" BTC Price History")

    time_range = st.select_slider(
        "Select Range",
        options=["1M", "3M", "6M", "1Y", "5Y"],
        value="1Y"
    )

    range_map = {
        "1M":  30,
        "3M":  90,
        "6M":  180,
        "1Y":  365,
        "5Y":  len(df)
    }

    days = range_map[time_range]

    #  Fix: load fresh with date index 
    chart_source        = pd.read_csv("data/price_data.csv")
    chart_source["Date"] = pd.to_datetime(chart_source["Date"])
    chart_source         = chart_source.set_index("Date")
    chart_source         = chart_source[["Close"]].tail(days).copy()
    chart_source.columns = ["BTC Price (USD)"]

    st.line_chart(
        chart_source,
        color               = "#2d5be3",
        use_container_width = True
    )

    st.divider()

    # Price Metrics 
    st.subheader(" Bitcoin Price")

    st.metric(
        label = "Current BTC Price",
        value = f"${analysis['current_price']:,.2f}"
    )

    lstm_delta = analysis["lstm_price"] - analysis["current_price"]
    st.metric(
        label       = "LSTM Predicted Tomorrow",
        value       = f"${analysis['lstm_price']:,.2f}",
        delta       = round(lstm_delta, 2),
        delta_color = "normal"
    )

    st.divider()

    #  Sentiment 
    st.subheader(" Market Sentiment")
    score = analysis["sentiment_score"]

    if score > 0.2:
        sentiment_label = "Positive 📈"
    elif score < -0.2:
        sentiment_label = "Negative 📉"
    else:
        sentiment_label = "Neutral ➡️"

    st.metric(
        label = "Market Mood",
        value = sentiment_label,
        delta = f"Score: {score:.2f}"
    )

    st.divider()

    # RL Recommendation 
    st.subheader("🤖 AI Recommendation")
    if analysis["action_label"] == "BUY":
        st.success(f"### {analysis['action_label']} ✅")
    elif analysis["action_label"] == "SELL":
        st.error(f"### {analysis['action_label']} 🔴")
    else:
        st.warning(f"### {analysis['action_label']} ⚠️")

    st.divider()

    #  Navigation Buttons ─
    st.subheader(" Navigation")

    if st.button(" Chat Advisor"):
        st.session_state.page = "chat"
        st.rerun()

    if st.button(" Technical Analysis & Backtest"):
        st.session_state.page = "analysis"
        st.rerun()

    st.divider()
    st.caption("⚠️ Not financial advice. "
               "Crypto investing carries significant risk.")

    if st.button(" Refresh Analysis"):
        st.cache_data.clear()
        st.rerun()


# Page: Chat 
if st.session_state.page == "chat":

    st.title(" Chat with your Crypto Advisor")
    st.caption("Ask me anything about Bitcoin — "
               "I will explain it in simple terms!")
    st.divider()

    # Quick question buttons
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Should I buy Bitcoin today?"):
            st.session_state.user_input = "Should I buy Bitcoin today?"
    with col2:
        if st.button("Why is the market sentiment negative?"):
            st.session_state.user_input = "Why is the market sentiment negative?"
    with col3:
        if st.button("What is the predicted price tomorrow?"):
            st.session_state.user_input = "What is the predicted price tomorrow?"

    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Build RAG context
    context = build_context(
        analysis["current_price"],
        analysis["lstm_price"],
        analysis["sentiment_score"],
        analysis["action_label"]
    )

    # Chat input
    prompt = st.chat_input("Ask me about crypto...")

    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = ""

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({
            "role": "user", "content": prompt
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = st.session_state.messages[:-1]
                response     = ask_advisor(prompt, context, chat_history)
                response     = format_response(response)
                st.write(response)

        st.session_state.messages.append({
            "role": "assistant", "content": response
        })


#  Page: Analysis 
elif st.session_state.page == "analysis":

    st.title("📊 Technical Analysis & Backtest Results")
    st.caption("Detailed model performance and backtesting metrics")
    st.divider()

    # Two tabs on this page
    tab1, tab2 = st.tabs([
        "🔬 Model Comparison",
        "📈 Backtest Results"
    ])

    # Tab 1: Model Comparison
    with tab1:
        st.subheader("🔬 LSTM vs Transformer Comparison")
        st.caption("Both models trained on 5 years of Bitcoin data")

        col1, col2 = st.columns(2)

        # LSTM Column
        with col1:
            st.markdown("### 🟢 LSTM (Selected)")
            st.metric("MAE",                  "$3,035.88")
            st.metric("RMSE",                 "$3,899.90")
            st.metric("Directional Accuracy", "50.1%")
            st.metric("Next Day Prediction",
                      f"${analysis['lstm_price']:,.2f}")
            st.success("✅ Selected as primary model")

        # Transformer Column
        with col2:
            st.markdown("### 🔴 Transformer")
            st.metric("MAE",                  "$19,801.11")
            st.metric("RMSE",                 "$22,701.66")
            st.metric("Directional Accuracy", "49.0%")
            st.metric("Next Day Prediction",
                      f"${analysis['transformer_price']:,.2f}")
            st.error("❌ Not selected — higher error")

        st.divider()
        st.markdown("###  Why LSTM was Selected")
        st.info("""
        The Transformer model significantly underperformed the LSTM
        across all metrics, achieving a MAE of $19,801.11 compared
        to the LSTM's $3,035.88. This is consistent with findings
        in the literature, Transformer models require significantly
        larger datasets to generalise effectively. With only 1,826
        daily samples available, the Transformer exhibited clear
        signs of overfitting while the LSTM's recurrent architecture
        proved better suited to this dataset size. The LSTM was
        therefore selected as the primary forecasting component
        for the system.
        """)

        if os.path.exists("data/model_comparison.png"):
            st.divider()
            st.markdown("###  Prediction Chart")
            st.image("data/model_comparison.png",
                     caption="Actual vs LSTM vs Transformer Predictions",
                     use_container_width=True)

    # Tab 2: Backtest Results
    with tab2:
        st.subheader(" RL Agent Backtest Results")
        st.caption("Simulated trading over 5 years of Bitcoin data")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cumulative Return",
                      f"{analysis['cum_return']:+.2f}%")
        with col2:
            st.metric("Sharpe Ratio",
                      f"{analysis['sharpe']:.4f}")
        with col3:
            st.metric("Max Drawdown",
                      f"{analysis['max_dd']:.2f}%")
        with col4:
            st.metric("Buy & Hold Return",
                      f"{analysis['bh_return']:+.2f}%")

        st.divider()

        # Trading activity
        history_df = analysis["history_df"]
        buys       = len(history_df[history_df["action"] == "BUY"])
        sells      = len(history_df[history_df["action"] == "SELL"])
        holds      = len(history_df[history_df["action"] == "HOLD"])

        st.markdown("###  Trading Activity")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BUY Actions",  buys)
        with col2:
            st.metric("SELL Actions", sells)
        with col3:
            st.metric("HOLD Actions", holds)

        st.divider()

        st.markdown("### Evaluation Notes")
        st.warning("""
        The RL agent achieved a cumulative return of +186.18% over
        the 5-year backtest period, significantly outperforming the
        buy and hold benchmark of +83.08% by +103.11%. The Sharpe
        ratio of 0.5669 indicates acceptable risk-adjusted returns.
        The maximum drawdown of -35.71% reflects the inherent
        volatility of the Bitcoin market. The agent successfully
        executed 745 BUY, 756 SELL and 265 HOLD actions across
        1,766 trading days.
        """)

        if os.path.exists("data/backtest_results.png"):
            st.divider()
            st.markdown("###  Portfolio vs Buy & Hold")
            st.image("data/backtest_results.png",
                     caption="RL Agent Portfolio Value vs Buy & Hold",
                     use_container_width=True)

        if os.path.exists("data/lstm_predictions.png"):
            st.markdown("###  LSTM Predictions")
            st.image("data/lstm_predictions.png",
                     caption="LSTM Predicted vs Actual Bitcoin Price",
                     use_container_width=True)


# Footer 
st.divider()
st.caption("🤖 Crypto Advisor Bot | "
           "LSTM · Transformer · FinBERT · PPO · LLaMA 3.2 · Streamlit")