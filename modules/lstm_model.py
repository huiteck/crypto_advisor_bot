import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Prepare Data
def prepare_data(df, lookback=60):

    # Use only the closing price
    prices = df[["Close"]].values

    # Scale prices to be between 0 and 1
    # LSTM works better with small numbers
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    # Split into 80% training, 20% testing
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Data prepared. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


# Build LSTM Model
def build_model(input_shape):
   
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # randomly turns off 20% of neurons to prevent overfitting
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)  # output = single price prediction
    ])

    model.compile(optimizer="adam", loss="mse")
    print("LSTM model built.")
    print(model.summary())
    return model


# Train Model 
def train_model(model, X_train, y_train):

    # Early stopping stops training if model stops improving
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Training LSTM model... (this may take a few minutes)")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    print("Model trained and saved to models/lstm_model.keras")
    return history


# Evaluate & Plot 
def evaluate_model(model, X_test, y_test, scaler):
   
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test)

    # Calculate metrics
    mae  = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    # Directional accuracy
    pred_dir   = np.diff(predictions.flatten())
    actual_dir = np.diff(actual.flatten())
    dir_acc    = np.mean(np.sign(pred_dir) == np.sign(actual_dir)) * 100

    print(f"\n LSTM Evaluation:")
    print(f"   MAE:                  ${mae:,.2f}")
    print(f"   RMSE:                 ${rmse:,.2f}")
    print(f"   Directional Accuracy: {dir_acc:.1f}%")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(actual,      label="Actual Price",    color="blue")
    plt.plot(predictions, label="Predicted Price", color="orange")
    plt.title("Bitcoin Price: Actual vs Predicted")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/lstm_predictions.png")
    plt.show()
    print(" Chart saved to data/lstm_predictions.png")

    return predictions, actual


# Predict Next Day Price
def predict_next_price(df, scaler, model, lookback=60):

    prices = df[["Close"]].values
    scaled = scaler.transform(prices)

    # Take the last 60 days as input
    last_sequence = scaled[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # Predict and convert back to real price
    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)

    predicted_price = float(prediction[0][0])
    print(f"Predicted next day BTC price: ${predicted_price:,.2f}")
    return predicted_price


# Test
if __name__ == "__main__":
    # Download fresh price data (fixes the BTC-USD header issue)
    import sys
    import os
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ))
    from modules.data_collector import get_price_data

    df = get_price_data()

    # Drop any non-numeric rows just in case
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df["Close"] = df["Close"].astype(float)
    df = df.reset_index(drop=True)

    # Prepare sequences
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Build model
    model = build_model((X_train.shape[1], 1))

    # Train model
    history = train_model(model, X_train, y_train)

    # Evaluate model
    predictions, actual = evaluate_model(model, X_test, y_test, scaler)

    # Predict next day
    next_price = predict_next_price(df, scaler, model)
    print(f"Tomorrow's predicted BTC price: ${next_price:,.2f}")