import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ── VADER ──────────────────────────────────────────────────────
def run_vader(df):
    print("\n[1/3] Running VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if not isinstance(text, str) or len(text) == 0:
            return 'neutral', 0.0
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound

    results = df['model_text'].apply(get_vader_sentiment)
    df['vader_sentiment'] = results.apply(lambda x: x[0])
    df['vader_score'] = results.apply(lambda x: x[1])

    print("  VADER distribution:")
    print(df['vader_sentiment'].value_counts())
    return df

# ── LOGISTIC REGRESSION ────────────────────────────────────────
def run_logistic_regression(df):
    print("\n[2/3] Running Logistic Regression...")

    # Use VADER labels as training signal
    # (semi-supervised: VADER labels -> LR learns TF-IDF features)
    labeled = df[df['vader_sentiment'] != 'neutral'].copy()
    print(f"  Training on {len(labeled)} non-neutral samples...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(labeled['clean_text'])
    y = labeled['vader_sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Predict on full dataset
    X_full = vectorizer.transform(df['clean_text'])
    df['lr_sentiment'] = model.predict(X_full)
    df['lr_confidence'] = model.predict_proba(X_full).max(axis=1)

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return df, metrics

# ── DISTILBERT ─────────────────────────────────────────────────
def run_distilbert(df):
    print("\n[3/3] Running DistilBERT sentiment analysis...")
    print("  Loading model (first time may take a minute)...")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU
        truncation=True,
        max_length=512
    )

    results = []
    texts = df['model_text'].fillna('').tolist()
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Truncate long texts
        batch = [t[:512] if isinstance(t, str) else '' for t in batch]
        try:
            preds = sentiment_pipeline(batch)
            results.extend(preds)
        except Exception as e:
            print(f"  Batch error at {i}: {e}")
            results.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))

        if i % 500 == 0:
            print(f"  Processed {i}/{len(texts)}...")

    df['distilbert_sentiment'] = [r['label'].lower() for r in results]
    df['distilbert_score'] = [r['score'] for r in results]

    # Normalize labels
    df['distilbert_sentiment'] = df['distilbert_sentiment'].replace({
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral'
    })

    print("  DistilBERT distribution:")
    print(df['distilbert_sentiment'].value_counts())

    return df

# ── MODEL AGREEMENT ANALYSIS ───────────────────────────────────
def analyze_agreement(df):
    print("\nAnalyzing model agreement...")

    def check_agreement(row):
        sentiments = [row['vader_sentiment'], row['lr_sentiment'], row['distilbert_sentiment']]
        sentiments = [s for s in sentiments if s != 'neutral']
        if len(set(sentiments)) == 1:
            return 'all_agree'
        elif len(sentiments) == 0:
            return 'all_neutral'
        else:
            return 'disagree'

    df['model_agreement'] = df.apply(check_agreement, axis=1)

    print("  Agreement stats:")
    print(df['model_agreement'].value_counts())

    # Find interesting disagreements
    disagreements = df[df['model_agreement'] == 'disagree'][
        ['title', 'vader_sentiment', 'lr_sentiment', 'distilbert_sentiment']
    ].head(10)

    print("\n  Sample disagreements (most interesting for your report):")
    print(disagreements.to_string())

    return df

# ── MAIN ───────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 50)

    df = pd.read_csv("data/processed_data.csv")
    print(f"Loaded {len(df)} rows")

    df = run_vader(df)
    df, lr_metrics = run_logistic_regression(df)
    df = run_distilbert(df)
    df = analyze_agreement(df)

    os.makedirs("models", exist_ok=True)
    with open("models/lr_metrics.json", "w") as f:
        json.dump(lr_metrics, f)

    df.to_csv("data/sentiment_results.csv", index=False)

    print("\n" + "=" * 50)
    print("Done! Saved to data/sentiment_results.csv")
    print(f"\nFinal sentiment overview:")
    print("\nVADER:")
    print(df['vader_sentiment'].value_counts())
    print("\nLogistic Regression:")
    print(df['lr_sentiment'].value_counts())
    print("\nDistilBERT:")
    print(df['distilbert_sentiment'].value_counts())

if __name__ == "__main__":
    main()
