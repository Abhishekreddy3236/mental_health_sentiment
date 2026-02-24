# 🧠 Mental Health Sentiment Monitor
### Social Media Big Data Analytics for Public Sentiment Monitoring

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Spark](https://img.shields.io/badge/Apache%20Spark-4.0.2-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

A complete end-to-end NLP and Big Data pipeline that analyzes sentiment and emotions in Reddit mental health communities using three different AI models, topic modeling, real-time streaming, and distributed big data processing via Apache Spark.

---

## 📌 Project Overview

This project builds a multi-model sentiment analysis system applied to Reddit mental health communities. It goes beyond binary sentiment by including emotion classification, topic modeling, and a live real-time data stream — all presented in an interactive Streamlit dashboard.

### Key Features
- **3-Model Sentiment Analysis** — VADER, Logistic Regression, and DistilBERT compared side by side
- **Emotion Classification** — 6 emotions (sadness, joy, fear, anger, love, surprise) across 5 communities
- **Topic Modeling** — BERTopic discovers latent themes without supervision
- **Big Data Processing** — Apache Spark 4.0 processes 1.6 million tweets
- **Real-Time Streaming** — Live HackerNews sentiment monitoring via public API
- **Interactive Dashboard** — 11-section Streamlit dashboard with filters and insights

---

## 🗂️ Project Structure

```
mental_health_sentiment/
│
├── data/
│   ├── reddit_mental_health.csv      # Raw Kaggle dataset (5,957 posts)
│   ├── processed_data.csv            # After preprocessing (4,620 posts)
│   ├── sentiment_results.csv         # After 3-model sentiment analysis
│   ├── emotion_results.csv           # After emotion classification
│   ├── final_results.csv             # Complete dataset with all features
│   └── live_stream.csv               # HackerNews real-time stream data
│
├── models/
│   ├── lr_metrics.json               # Logistic Regression performance metrics
│   ├── topic_info.csv                # BERTopic topic information
│   ├── topic_words.json              # Top words per topic
│   ├── spark_metrics.json            # PySpark MLlib metrics
│   ├── bigdata_metrics.json          # Sentiment140 big data metrics
│   └── stream_summary.json           # Live stream summary statistics
│
├── 2_preprocess.py                   # Text cleaning and feature engineering
├── 3_sentiment_models.py             # VADER + Logistic Regression + DistilBERT
├── 4_emotion_classifier.py           # 6-emotion classification
├── 5_topic_modeling.py               # BERTopic topic discovery
├── 6_dashboard.py                    # Streamlit interactive dashboard
├── 7_spark_analysis.py               # PySpark SQL + MLlib on Reddit data
├── 8_spark_bigdata.py                # PySpark on Sentiment140 (1.6M tweets)
├── 9_realtime_stream.py              # HackerNews live sentiment stream
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9 |
| Dashboard | Streamlit + Plotly |
| NLP Models | VADER, Scikit-learn, HuggingFace DistilBERT |
| Topic Modeling | BERTopic + SentenceTransformers |
| Big Data | Apache Spark 4.0 + Spark MLlib |
| Real-Time | HackerNews Firebase API |
| Data Source | Kaggle Reddit Mental Health Dataset |
| Big Data Source | Sentiment140 (1.6M tweets) |

---

## 📊 Data Sources

### 1. Reddit Mental Health Dataset (Primary)
- **Source:** Kaggle — Reddit Mental Health Dataset
- **Size:** 4,620 posts after preprocessing
- **Communities:** r/depression, r/anxiety, r/mentalhealth, r/SocialAnxiety, r/Mindfulness
- **Used for:** Full NLP pipeline — sentiment, emotion, topic modeling

### 2. Sentiment140 (Big Data Validation)
- **Source:** Kaggle — Sentiment140
- **Size:** 1,600,000 tweets
- **Used for:** Demonstrating Apache Spark at genuine big data scale
- **Note:** Not included in repo due to file size (230MB) — download separately

### 3. HackerNews Live API
- **Source:** HackerNews Firebase API (hacker-news.firebaseio.com)
- **Size:** 200 live posts per run
- **Used for:** Real-time sentiment monitoring demonstration
- **Note:** Free, no authentication required

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9
- Java 17 (required for PySpark)
- macOS / Linux

### Install Java 17 (macOS)
```bash
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH=$JAVA_HOME/bin:$PATH
```

### Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run the Full Pipeline (first time only)
```bash
source venv/bin/activate
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH=$JAVA_HOME/bin:$PATH

python3 2_preprocess.py          # Step 1: Clean data
python3 3_sentiment_models.py    # Step 2: Sentiment analysis
python3 4_emotion_classifier.py  # Step 3: Emotion classification
python3 5_topic_modeling.py      # Step 4: Topic modeling
python3 7_spark_analysis.py      # Step 5: PySpark on Reddit data
python3 8_spark_bigdata.py       # Step 6: PySpark on 1.6M tweets
python3 9_realtime_stream.py     # Step 7: Collect live HackerNews posts
```

### Launch the Dashboard
```bash
python3 -m streamlit run 6_dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Models Used

### VADER
- Rule-based lexicon approach, no training required
- Fast but lacks contextual understanding
- **Result:** 59% negative, 37% positive, 4% neutral

### Logistic Regression (TF-IDF + Scikit-learn)
- TF-IDF features (10,000 features, 1-2 grams)
- Semi-supervised — trained on VADER labels
- **Accuracy:** 75.93%

### DistilBERT (Transformer)
- Pre-trained transformer fine-tuned on SST-2
- Understands full sentence context and nuance
- **Result:** 87% negative — correctly captures emotional complexity

### Why Three Models?
The disagreement analysis is the core academic contribution. Example: *"I'm desperate for a friend and to feel loved by someone"* — VADER and LR classify as **positive** (detected "friend" and "loved"), DistilBERT classifies as **negative** (understood desperation). This demonstrates why context-aware models are essential for mental health text.

---

## 📈 Key Findings

| Finding | Detail |
|---|---|
| Depression negativity | 92.6% negative — highest of all communities |
| Mindfulness surprise | Dominant emotion is **Fear (35.7%)**, not Joy |
| DistilBERT vs VADER | 87% vs 59% negative — context matters |
| Spark MLlib accuracy | 86.68% — outperforms scikit-learn LR (75.93%) |
| Big data scale | 1,280,209 training samples processed in 29.7 seconds |
| Topic: Loneliness | 1,273 posts, 89% negative — largest meaningful cluster |

---

## ⚡ Big Data Component

**Reddit Analytics (7_spark_analysis.py)**
- Spark SQL aggregations on community-level sentiment
- MLlib pipeline: 86.68% accuracy

**Sentiment140 Scale Validation (8_spark_bigdata.py)**
- 1,600,000 tweets loaded in **7.4 seconds**
- 1,280,209 training samples processed in **29.7 seconds**
- 76.72% accuracy on 319,791 test records

---

## 📡 Real-Time Streaming

`9_realtime_stream.py` connects to the HackerNews public API and streams live posts, scoring each with VADER in real time.

**Live collection results:** 200 posts · 56.5% Neutral · 23.5% Positive · 20.0% Negative

---

## 📁 Note on Large Files

`data/sentiment140.csv` (~230MB) is excluded from this repo. Download from Kaggle, place in `data/` folder, and run `8_spark_bigdata.py`.

All other data files are generated by running the pipeline scripts in order.

---

## 👨‍💻 Author

**Abhishek Reddy Kotha**
Built with Python 3.9 · Apache Spark 4.0 · HuggingFace Transformers · Streamlit
