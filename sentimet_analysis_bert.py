"""
Mini Project: Sentiment Analysis using BERT
Description:
This project demonstrates the use of a pretrained Transformer model (DistilBERT)
for classifying IMDb-like movie reviews as Positive or Negative.
"""
# Import Libraries

from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Pretrained Model
def load_model():
    """Load the pretrained BERT sentiment analysis pipeline."""
    print("Loading pretrained BERT model...")
    model = pipeline("sentiment-analysis")
    print("Model loaded successfully!\n")
    return model

# Step 2: Analyze Reviews
def analyze_reviews(model, reviews):
    """Run sentiment analysis on a list of reviews."""
    print("Analyzing sample IMDb reviews...\n")
    results = model(reviews)
    df = pd.DataFrame({
        "Review": reviews,
        "Predicted Sentiment": [r['label'] for r in results],
        "Confidence": [round(r['score'], 3) for r in results]
    })
    return df


# Step 3: Visualize Results
def visualize_results(df):
    """Plot sentiment distribution for the sample dataset."""
    sentiment_counts = df['Predicted Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Sentiment Distribution of Sample Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()


# Step 4: Main Function
def main():
    sentiment_analyzer = load_model()

    # Sample IMDb-like movie reviews
    reviews = [
        "I absolutely loved this movie! The acting was fantastic and the story was beautiful.",
        "It was a waste of time. The plot was boring and predictable.",
        "An average movie, not too bad but nothing special either.",
        "The direction and cinematography were breathtaking. Highly recommend it!",
        "Terrible movie. I walked out halfway through."
    ]
    
    df = analyze_reviews(sentiment_analyzer, reviews)
    print("\nSentiment Analysis Results:\n")
    print(df)
    
    # Save results to CSV
    df.to_csv("sentiment_results.csv", index=False)
    print("\n✅ Results saved to sentiment_results.csv")
    
    # Visualize the results
    visualize_results(df)

    # Optional: Take user input
    user_review = input("\nEnter your own movie review: ")
    print(sentiment_analyzer(user_review))


if __name__ == "__main__":
    main()
