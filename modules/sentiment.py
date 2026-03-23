from transformers import pipeline
import pandas as pd
import os

# Load FinBERT Sentiment Model
def load_sentiment_model():

    print("Loading FinBERT sentiment model...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )
    print("Sentiment model loaded.")
    return sentiment_model


# Analyse Headlines
def analyse_headlines(headlines, sentiment_model):

    print(f"Analysing {len(headlines)} headlines...")

    results = []
    for headline in headlines:
        # Truncate headline to 512 characters (model limit)
        headline = headline[:512]

        # Get sentiment prediction
        result = sentiment_model(headline)[0]
        label = result["label"]    # positive, neutral, negative
        score = result["score"]    # confidence 0 to 1

        results.append({
            "headline": headline,
            "label": label,
            "score": score
        })

    # Convert to dataframe for easy reading
    df = pd.DataFrame(results)

    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sentiment_results.csv", index=False)
    print("Sentiment results saved to data/sentiment_results.csv")

    return df


# Calculate Overall Sentiment Score
def get_overall_sentiment(df):

    score_map = {
        "positive":  1,
        "neutral":   0,
        "negative": -1
    }

    # Map labels to numbers
    df["numeric_score"] = df["label"].map(score_map)

    # Calculate average
    overall = df["numeric_score"].mean()

    # Count how many of each
    counts = df["label"].value_counts()
    positive_count = counts.get("positive", 0)
    neutral_count  = counts.get("neutral",  0)
    negative_count = counts.get("negative", 0)

    print(f"\n Sentiment Summary:")
    print(f"   Positive headlines: {positive_count}")
    print(f"   Neutral  headlines: {neutral_count}")
    print(f"   Negative headlines: {negative_count}")
    print(f"   Overall score:      {overall:.2f} (-1 to +1)")

    if overall > 0.2:
        mood = "Positive - Market mood is bullish"
    elif overall < -0.2:
        mood = "Negative - Market mood is bearish"
    else:
        mood = "Neutral  - Market mood is uncertain"

    print(f"   Market mood:        {mood}")
    return overall


# Test
if __name__ == "__main__":
    # Load news headlines from data collector
    import sys
    sys.path.append("..")
    from modules.data_collector import get_news_headlines

    # Get latest headlines
    headlines = get_news_headlines()

    # Load sentiment model
    model = load_sentiment_model()

    # Analyse headlines
    df = analyse_headlines(headlines, model)

    # Print each headline with its sentiment
    print("\n Individual Headline Results:")
    print("-" * 60)
    for _, row in df.iterrows():
        emoji = "🟢" if row["label"] == "positive" else \
                "🔴" if row["label"] == "negative" else "🟡"
        print(f"{emoji} {row['label'].upper():8} | {row['headline'][:60]}...")

    # Get overall score
    overall_score = get_overall_sentiment(df)
    print(f"\n Final sentiment score: {overall_score:.2f}")