import yfinance as yf
import pandas as pd
import feedparser
import os

# Download BTC price
def get_price_data(symbol="BTC-USD", period="5y"):
    
    print(f"Downloading price data for {symbol}...")
    df = yf.download(symbol, period=period, auto_adjust=True)

    df.columns = df.columns.get_level_values(0)  # flatten column names
    df.dropna(inplace=True)  # remove any empty rows

    # Save to CSV 
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/price_data.csv")
    print(f"Price data saved! {len(df)} rows downloaded.")
    return df


# Scrape Crypto News Headlines 
def get_news_headlines():
    
    # Fetches latest crypto news from CoinDesk RSS feed.
    
    print("Fetching latest crypto news...")
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries[:20]:  # get latest 20 headlines
        headlines.append(entry.title)

    print(f"Fetched {len(headlines)} news headlines.")
    return headlines


# Load Saved Price Data
def load_price_data():

    df = pd.read_csv("data/price_data.csv")
    print(f"Loaded price data. {len(df)} rows.")
    return df


# Test
if __name__ == "__main__":
    # Download and save price data
    df = get_price_data()
    print(df.tail())  # show last 5 rows

    # Fetch news
    headlines = get_news_headlines()
    for i, h in enumerate(headlines):
        print(f"{i+1}. {h}")