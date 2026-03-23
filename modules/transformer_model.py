import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Model, load_model as tf_load_model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


# Transformer
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):

    # Multi-head self attention
    attention_output = MultiHeadAttention(
        num_heads = num_heads,
        key_dim   = inputs.shape[-1] // num_heads
    )(inputs, inputs)

    attention_output = Dropout(dropout_rate)(attention_output)

    # Add & Norm (residual connection)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed forward network
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)

    # Add & Norm (residual connection)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    return out2


# Building the model
def build_transformer(lookback=60, num_heads=4, ff_dim=64,
                      num_blocks=2, dropout_rate=0.1):

    inputs = Input(shape=(lookback, 1))

    # Project input to a higher dimension for attention
    x = Dense(64)(inputs)

    # Stack multiple transformer blocks
    for _ in range(num_blocks):
        x = transformer_block(x, num_heads, ff_dim, dropout_rate)

    # Pool across the sequence dimension
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    print("Transformer model built.")
    model.summary()
    return model


# Prepare Data
def prepare_data(df, lookback=60):

    prices = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Data prepared. Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


# Training transformer
def train_transformer(model, X_train, y_train):

    early_stop = EarlyStopping(
        monitor             = "val_loss",
        patience            = 5,
        restore_best_weights = True
    )

    print("Training Transformer model... (this may take a few minutes)")
    history = model.fit(
        X_train, y_train,
        epochs           = 50,
        batch_size       = 32,
        validation_split = 0.1,
        callbacks        = [early_stop],
        verbose          = 1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/transformer_model.keras")
    print("Transformer model saved to models/transformer_model.keras")
    return history


# Evaluate
def evaluate_transformer(model, X_test, y_test, scaler):

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test)

    mae  = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    # Directional accuracy
    pred_dir   = np.diff(predictions.flatten())
    actual_dir = np.diff(actual.flatten())
    dir_acc    = np.mean(np.sign(pred_dir) == np.sign(actual_dir)) * 100

    print(f"\n Transformer Evaluation:")
    print(f"   MAE:                  ${mae:,.2f}")
    print(f"   RMSE:                 ${rmse:,.2f}")
    print(f"   Directional Accuracy: {dir_acc:.1f}%")

    return predictions, actual, mae, rmse, dir_acc


# Predict next price
def predict_next_price(df, scaler, model, lookback=60):

    prices = df[["Close"]].values
    scaled = scaler.transform(prices)

    last_sequence = scaled[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction        = scaler.inverse_transform(prediction_scaled)
    predicted_price   = float(prediction[0][0])

    print(f"Transformer predicted next day BTC: ${predicted_price:,.2f}")
    return predicted_price


# Comparison of LSTM and Transformer
def compare_models(lstm_preds, transformer_preds, actual,
                   lstm_mae, transformer_mae,
                   lstm_rmse, transformer_rmse,
                   lstm_dir, transformer_dir):

    # Plotting graph for lstm vs transformer

    print("\n" + "=" * 55)
    print("       MODEL COMPARISON: LSTM vs TRANSFORMER")
    print("=" * 55)
    print(f"{'Metric':<25} {'LSTM':>12} {'Transformer':>15}")
    print("-" * 55)
    print(f"{'MAE ($)':<25} {'${:,.2f}'.format(lstm_mae):>12} "
          f"{'${:,.2f}'.format(transformer_mae):>15}")
    print(f"{'RMSE ($)':<25} {'${:,.2f}'.format(lstm_rmse):>12} "
          f"{'${:,.2f}'.format(transformer_rmse):>15}")
    print(f"{'Directional Acc (%)':<25} {'{:.1f}%'.format(lstm_dir):>12} "
          f"{'{:.1f}%'.format(transformer_dir):>15}")
    print("=" * 55)

    winner_mae  = "LSTM" if lstm_mae  < transformer_mae  else "Transformer"
    winner_rmse = "LSTM" if lstm_rmse < transformer_rmse else "Transformer"
    winner_dir  = "LSTM" if lstm_dir  > transformer_dir  else "Transformer"
    print(f"\n Best MAE:                  {winner_mae}")
    print(f"Best RMSE:                 {winner_rmse}")
    print(f"Best Directional Accuracy: {winner_dir}")

    # Plot comparison chart

    plt.figure(figsize=(14, 6))
    plt.plot(actual.flatten(),           label="Actual Price",
             color="blue",  linewidth=2)
    plt.plot(lstm_preds.flatten(),        label="LSTM Predicted",
             color="orange", linewidth=1.5, linestyle="--")
    plt.plot(transformer_preds.flatten(), label="Transformer Predicted",
             color="green",  linewidth=1.5, linestyle=":")
    plt.title("Bitcoin Price: Actual vs LSTM vs Transformer")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/model_comparison.png")
    plt.show()
    print("Comparison chart saved to data/model_comparison.png")

    return winner_mae


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ))

    from modules.data_collector import get_price_data
    from tensorflow.keras.models import load_model as tf_load_model

    # Load data
    print(" Loading price data...")
    df = get_price_data()
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df["Close"] = df["Close"].astype(float)
    df = df.reset_index(drop=True)

    # Train Transformer
    print("\n Training Transformer model...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    transformer = build_transformer()
    train_transformer(transformer, X_train, y_train)

    # Evaluate Transformer
    t_preds, actual, t_mae, t_rmse, t_dir = evaluate_transformer(
        transformer, X_test, y_test, scaler
    )

    # Load and evaluate LSTM for comparison
    print("\n Loading LSTM for comparison...")
    lstm_model = tf_load_model("models/lstm_model.keras")

    # Get LSTM predictions on same test set
    lstm_preds_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled)

    # Compute LSTM metrics
    lstm_mae  = float(np.mean(np.abs(lstm_preds - actual)))
    lstm_rmse = float(np.sqrt(np.mean((lstm_preds - actual) ** 2)))
    l_dir_p   = np.diff(lstm_preds.flatten())
    l_dir_a   = np.diff(actual.flatten())
    lstm_dir  = float(np.mean(np.sign(l_dir_p) == np.sign(l_dir_a)) * 100)

    print(f"\n LSTM Metrics (for comparison):")
    print(f"   MAE:                  ${lstm_mae:,.2f}")
    print(f"   RMSE:                 ${lstm_rmse:,.2f}")
    print(f"   Directional Accuracy: {lstm_dir:.1f}%")

    # Compare Both model
    compare_models(
        lstm_preds, t_preds, actual,
        lstm_mae,  t_mae,
        lstm_rmse, t_rmse,
        lstm_dir,  t_dir
    )

    # Next day predictions for both model
    print("\n Next day predictions:")
    prices_scaled = scaler.transform(df[["Close"]].values)
    last_seq      = np.expand_dims(prices_scaled[-60:], axis=0)

    lstm_next        = float(scaler.inverse_transform(
        lstm_model.predict(last_seq, verbose=0))[0][0])
    transformer_next = predict_next_price(df, scaler, transformer)
    avg_prediction   = (lstm_next + transformer_next) / 2

    print(f"   LSTM predicted:        ${lstm_next:,.2f}")
    print(f"   Transformer predicted: ${transformer_next:,.2f}")
    print(f"   Ensemble average:      ${avg_prediction:,.2f}")