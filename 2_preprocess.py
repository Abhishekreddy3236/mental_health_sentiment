import pandas as pd
import re
import nltk
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LABEL_MAP = {
    0: "Anxiety",
    1: "Depression",
    2: "Mental Health",
    3: "Social Anxiety",
    4: "Mindfulness"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'/r/\w+', '', text)
    text = re.sub(r'/u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    keep_words = {'not', 'no', 'never', 'nothing', 'neither', 'nobody', 'nowhere', 'nor', 'cannot'}
    stop_words = stop_words - keep_words
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(filtered)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmatized)

def preprocess_pipeline(df):
    print("Step 1: Dropping nulls and duplicates...")
    df = df.dropna(subset=['title'])
    df = df.drop_duplicates(subset=['title'])
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    print(f"  -> {len(df)} rows remaining")

    print("Step 2: Combining title and text columns...")
    df['text'] = df['text'].fillna('')
    df['combined_text'] = df['title'] + ' ' + df['text']

    print("Step 3: Mapping community labels...")
    df['community'] = df['target'].map(LABEL_MAP)

    print("Step 4: Cleaning text...")
    df['clean_text'] = df['combined_text'].apply(clean_text)

    print("Step 5: Removing stopwords...")
    df['clean_text'] = df['clean_text'].apply(remove_stopwords)

    print("Step 6: Lemmatizing...")
    df['clean_text'] = df['clean_text'].apply(lemmatize_text)

    print("Step 7: Removing empty rows after cleaning...")
    df = df[df['clean_text'].str.len() > 10]
    print(f"  -> {len(df)} rows remaining after cleaning")

    print("Step 8: Creating transformer-friendly text...")
    df['model_text'] = df['combined_text'].apply(clean_text)

    print("Step 9: Adding text length features...")
    df['word_count'] = df['combined_text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['combined_text'].apply(lambda x: len(str(x)))

    return df

def main():
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    print("\nLoading data...")
    df = pd.read_csv("data/reddit_mental_health.csv")
    print(f"Loaded {len(df)} rows")

    df = preprocess_pipeline(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_data.csv", index=False)

    print("\n" + "=" * 50)
    print(f"Done! Saved to data/processed_data.csv")
    print(f"Final shape: {df.shape}")
    print(f"\nCommunity distribution:")
    print(df['community'].value_counts())
    print(f"\nSample cleaned text:")
    print(df[['title', 'clean_text', 'community']].head(3).to_string())

if __name__ == "__main__":
    main()
