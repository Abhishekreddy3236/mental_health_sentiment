# How to Run

cd ~/Downloads/mental_health_sentiment
source venv/bin/activate

# Run pipeline in order (already done, outputs saved):
python3 2_preprocess.py
python3 3_sentiment_models.py
python3 4_emotion_classifier.py
python3 5_topic_modeling.py
python3 7_spark_analysis.py

# Launch dashboard:
python3 -m streamlit run 6_dashboard.py
