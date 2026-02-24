import pandas as pd
import os
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

def run_emotion_classifier(df):
    print("Loading emotion classifier...")
    print("(This model detects: sadness, joy, love, anger, fear, surprise)")

    emotion_pipeline = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        device=-1,
        truncation=True,
        max_length=512
    )

    texts = df['model_text'].fillna('').tolist()
    results = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t[:512] if isinstance(t, str) else '' for t in batch]
        try:
            preds = emotion_pipeline(batch)
            results.extend(preds)
        except Exception as e:
            print(f"  Batch error at {i}: {e}")
            results.extend([{'label': 'sadness', 'score': 0.5}] * len(batch))

        if i % 500 == 0:
            print(f"  Processed {i}/{len(texts)}...")

    df['emotion'] = [r['label'] for r in results]
    df['emotion_score'] = [r['score'] for r in results]

    return df

def analyze_emotions(df):
    print("\nEmotion Distribution:")
    print(df['emotion'].value_counts())

    print("\nEmotion by Community:")
    emotion_community = pd.crosstab(df['community'], df['emotion'])
    print(emotion_community)

    print("\nMost common emotion per community:")
    for community in df['community'].unique():
        subset = df[df['community'] == community]
        top_emotion = subset['emotion'].value_counts().index[0]
        pct = subset['emotion'].value_counts().iloc[0] / len(subset) * 100
        print(f"  {community}: {top_emotion} ({pct:.1f}%)")

    print("\nSample posts by emotion:")
    for emotion in df['emotion'].unique():
        sample = df[df['emotion'] == emotion]['title'].iloc[0]
        print(f"\n  [{emotion.upper()}]")
        print(f"  {sample[:100]}...")

    return df

def main():
    print("=" * 50)
    print("EMOTION CLASSIFICATION PIPELINE")
    print("=" * 50)

    df = pd.read_csv("data/sentiment_results.csv")
    print(f"Loaded {len(df)} rows")

    df = run_emotion_classifier(df)
    df = analyze_emotions(df)

    df.to_csv("data/emotion_results.csv", index=False)

    print("\n" + "=" * 50)
    print("Done! Saved to data/emotion_results.csv")

if __name__ == "__main__":
    main()
