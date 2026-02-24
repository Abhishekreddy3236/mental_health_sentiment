import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

def run_topic_modeling(df):
    print("Preparing texts for topic modeling...")

    texts = df['clean_text'].fillna('').tolist()
    texts = [t for t in texts if len(t) > 20]
    print(f"  Using {len(texts)} texts")

    print("\nLoading sentence transformer (may take a moment)...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Building embeddings...")
    embeddings = sentence_model.encode(texts, show_progress_bar=True, batch_size=64)

    print("\nFitting BERTopic model...")
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=5000
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        min_topic_size=30,
        nr_topics=15,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    return topic_model, topics, probs, texts

def analyze_topics(topic_model, topics, texts, df):
    print("\n" + "=" * 50)
    print("TOPIC ANALYSIS")
    print("=" * 50)

    topic_info = topic_model.get_topic_info()
    print("\nAll discovered topics:")
    print(topic_info[['Topic', 'Count', 'Name']].to_string())

    print("\nTop words per topic:")
    topic_words = {}
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            word_list = [w[0] for w in words[:8]]
            topic_words[topic_id] = word_list
            print(f"\n  Topic {topic_id}: {', '.join(word_list)}")

    # Add topics back to dataframe
    texts_in_df = df['clean_text'].fillna('').tolist()
    topic_assignments = []
    text_set = {t: i for i, t in enumerate(texts)}

    topics_list = [-1] * len(df)
    text_idx = 0
    for i, text in enumerate(texts_in_df):
        if len(text) > 20:
            if text_idx < len(topics):
                topics_list[i] = topics[text_idx]
                text_idx += 1

    df['topic_id'] = topics_list

    # Map topic names
    topic_name_map = {}
    for _, row in topic_info.iterrows():
        topic_name_map[row['Topic']] = row['Name']
    df['topic_name'] = df['topic_id'].map(topic_name_map).fillna('outlier')

    print("\nTopic distribution:")
    print(df['topic_name'].value_counts().head(10))

    # Topic x Community analysis
    print("\nTopic x Community breakdown:")
    topic_community = pd.crosstab(df['topic_name'], df['community'])
    print(topic_community)

    # Topic x Sentiment analysis
    print("\nTop topics by sentiment:")
    topic_sentiment = pd.crosstab(df['topic_name'], df['distilbert_sentiment'])
    print(topic_sentiment)

    return df, topic_words, topic_info

def save_outputs(topic_model, topic_words, topic_info):
    os.makedirs("models", exist_ok=True)

    # Save topic words for dashboard
    with open("models/topic_words.json", "w") as f:
        json.dump(topic_words, f)

    # Save topic info
    topic_info.to_csv("models/topic_info.csv", index=False)

    print("\nSaved topic model outputs to models/")

def main():
    print("=" * 50)
    print("TOPIC MODELING PIPELINE")
    print("=" * 50)

    df = pd.read_csv("data/emotion_results.csv")
    print(f"Loaded {len(df)} rows")

    topic_model, topics, probs, texts = run_topic_modeling(df)
    df, topic_words, topic_info = analyze_topics(topic_model, topics, texts, df)
    save_outputs(topic_model, topic_words, topic_info)

    df.to_csv("data/final_results.csv", index=False)

    print("\n" + "=" * 50)
    print("Done! Saved to data/final_results.csv")
    print("All analysis complete. Ready to build dashboard.")

if __name__ == "__main__":
    main()
