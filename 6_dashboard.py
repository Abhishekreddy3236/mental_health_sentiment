import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4A90D9;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1E1E2E;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #4A90D9;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #4A90D9;
    }
    .insight-box {
        background: #1A2A3A;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #50C878;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/final_results.csv")
    with open("models/lr_metrics.json") as f:
        lr_metrics = json.load(f)
    return df, lr_metrics

df, lr_metrics = load_data()

# ── SIDEBAR ────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters")
communities = ['All'] + sorted(df['community'].unique().tolist())
selected_community = st.sidebar.selectbox("Select Community", communities)

sentiments = ['All', 'negative', 'positive', 'neutral']
selected_sentiment = st.sidebar.selectbox("Filter by DistilBERT Sentiment", sentiments)

emotions = ['All'] + sorted(df['emotion'].unique().tolist())
selected_emotion = st.sidebar.selectbox("Filter by Emotion", emotions)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.markdown(f"**Reddit Posts:** {len(df):,}")
st.sidebar.markdown(f"**Big Data (Sentiment140):** 1,600,000")
st.sidebar.markdown(f"**Live Stream Posts:** 200")
st.sidebar.markdown(f"**Communities:** 5")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Models Used")
st.sidebar.markdown("• VADER (Rule-based)")
st.sidebar.markdown("• Logistic Regression")
st.sidebar.markdown("• DistilBERT (Transformer)")
st.sidebar.markdown("• Emotion Classifier")
st.sidebar.markdown("• BERTopic")
st.sidebar.markdown("• Spark MLlib")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline")
st.sidebar.markdown("• Apache Spark 4.0")
st.sidebar.markdown("• HuggingFace Transformers")
st.sidebar.markdown("• HackerNews Live API")

# Apply filters
filtered_df = df.copy()
if selected_community != 'All':
    filtered_df = filtered_df[filtered_df['community'] == selected_community]
if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['distilbert_sentiment'] == selected_sentiment]
if selected_emotion != 'All':
    filtered_df = filtered_df[filtered_df['emotion'] == selected_emotion]

# ── HEADER ─────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Mental Health Sentiment Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Social Media Big Data Analytics · Reddit Mental Health Communities · 3-Model NLP Pipeline</div>', unsafe_allow_html=True)

# ── SECTION 1: KEY METRICS ─────────────────────────────────────
st.markdown('<div class="section-title">📈 Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

neg_pct = (filtered_df['distilbert_sentiment'] == 'negative').mean() * 100
pos_pct = (filtered_df['distilbert_sentiment'] == 'positive').mean() * 100
dom_emotion = filtered_df['emotion'].value_counts().index[0] if len(filtered_df) > 0 else 'N/A'
avg_vader = filtered_df['vader_score'].mean()
agreement_pct = (filtered_df['model_agreement'] == 'all_agree').mean() * 100 if 'model_agreement' in filtered_df.columns else 0

col1.metric("Total Posts", f"{len(filtered_df):,}")
col2.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta=f"{neg_pct-70:.1f}% vs avg")
col3.metric("Positive Sentiment", f"{pos_pct:.1f}%")
col4.metric("Dominant Emotion", dom_emotion.capitalize())
col5.metric("Model Agreement", f"{agreement_pct:.1f}%")

st.markdown("---")

# ── SECTION 2: SENTIMENT DISTRIBUTION ─────────────────────────
st.markdown('<div class="section-title">🎯 Sentiment Distribution Across Models</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    vader_counts = filtered_df['vader_sentiment'].value_counts().reset_index()
    vader_counts.columns = ['sentiment', 'count']
    fig_vader = px.pie(
        vader_counts, values='count', names='sentiment',
        title='VADER Sentiment',
        color='sentiment',
        color_discrete_map={'positive':'#50C878','negative':'#FF6B6B','neutral':'#4A90D9'},
        hole=0.4
    )
    fig_vader.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_vader, use_container_width=True)

with col2:
    distilbert_counts = filtered_df['distilbert_sentiment'].value_counts().reset_index()
    distilbert_counts.columns = ['sentiment', 'count']
    fig_db = px.pie(
        distilbert_counts, values='count', names='sentiment',
        title='DistilBERT Sentiment',
        color='sentiment',
        color_discrete_map={'positive':'#50C878','negative':'#FF6B6B','neutral':'#4A90D9'},
        hole=0.4
    )
    fig_db.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_db, use_container_width=True)

# Three model comparison bar chart
model_comparison = pd.DataFrame({
    'Model': ['VADER', 'VADER', 'Logistic Regression', 'Logistic Regression', 'DistilBERT', 'DistilBERT'],
    'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative'],
    'Count': [
        (filtered_df['vader_sentiment']=='positive').sum(),
        (filtered_df['vader_sentiment']=='negative').sum(),
        (filtered_df['lr_sentiment']=='positive').sum(),
        (filtered_df['lr_sentiment']=='negative').sum(),
        (filtered_df['distilbert_sentiment']=='positive').sum(),
        (filtered_df['distilbert_sentiment']=='negative').sum(),
    ]
})

fig_compare = px.bar(
    model_comparison, x='Model', y='Count', color='Sentiment',
    title='Three-Model Sentiment Comparison',
    color_discrete_map={'Positive':'#50C878','Negative':'#FF6B6B'},
    barmode='group'
)
fig_compare.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig_compare, use_container_width=True)

# Key insight
st.markdown("""
<div class="insight-box">
💡 <b>Key Insight:</b> DistilBERT detects significantly more negative sentiment than VADER 
because it understands context and nuance — e.g., "I'm desperate for a friend" is correctly 
classified as negative by DistilBERT but positive by VADER (which only saw the word "friend").
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 3: MODEL COMPARISON ───────────────────────────────
st.markdown('<div class="section-title">🔬 Model Performance Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.markdown("### VADER")
col1.markdown("**Type:** Rule-based lexicon")
col1.markdown("**Speed:** Instant")
col1.markdown("**Strength:** Simple, interpretable")
col1.markdown("**Weakness:** Misses context and sarcasm")

col2.markdown("### Logistic Regression")
col2.markdown(f"**Type:** Statistical ML (TF-IDF)")
col2.markdown(f"**Accuracy:** {lr_metrics['accuracy']:.2%}")
col2.markdown("**Strength:** Fast, explainable features")
col2.markdown("**Weakness:** Bag-of-words, no context")

col3.markdown("### DistilBERT")
col3.markdown("**Type:** Transformer (Deep Learning)")
col3.markdown("**Strength:** Context-aware, nuanced")
col3.markdown("**Weakness:** Slower, black box")
col3.markdown("**Best for:** Mental health text")

# Disagreement examples
st.markdown("#### Where Models Disagree (Most Informative Cases)")
disagreements = df[df['model_agreement'] == 'disagree'][
    ['title', 'vader_sentiment', 'lr_sentiment', 'distilbert_sentiment', 'community']
].head(8).reset_index(drop=True)
disagreements.columns = ['Post Title', 'VADER', 'Logistic Regression', 'DistilBERT', 'Community']
st.dataframe(disagreements, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Why disagreements matter:</b> These cases reveal the limits of each approach. 
Posts like <i>"Hope is just a form of self-harm"</i> are classified as positive by VADER 
(it detected "hope") but negative by DistilBERT (it understood the full meaning). 
This demonstrates why context-aware models are essential for mental health analysis.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 4: EMOTION ANALYSIS ───────────────────────────────
st.markdown('<div class="section-title">❤️ Emotion Classification Across Communities</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    emotion_counts = filtered_df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    fig_emotion = px.bar(
        emotion_counts, x='emotion', y='count',
        title='Overall Emotion Distribution',
        color='emotion',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_emotion.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

with col2:
    emotion_community = pd.crosstab(df['community'], df['emotion'])
    fig_heatmap = px.imshow(
        emotion_community,
        title='Emotion × Community Heatmap',
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig_heatmap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Stacked bar: emotion by community
emotion_comm_df = df.groupby(['community', 'emotion']).size().reset_index(name='count')
fig_stacked = px.bar(
    emotion_comm_df, x='community', y='count', color='emotion',
    title='Emotion Breakdown by Community',
    color_discrete_sequence=px.colors.qualitative.Set3,
    barmode='stack'
)
fig_stacked.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig_stacked, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Surprising Finding:</b> The Mindfulness community's dominant emotion is <b>Fear (35.7%)</b>, 
not joy. This reveals that people turn to mindfulness communities <i>because</i> they are anxious 
or afraid — not because they are already calm. Depression shows the highest sadness concentration (53.7%).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 5: TOPIC MODELING ─────────────────────────────────
st.markdown('<div class="section-title">🗺️ Topic Modeling (BERTopic)</div>', unsafe_allow_html=True)

topic_info = pd.read_csv("models/topic_info.csv")
valid_topics = topic_info[topic_info['Topic'] != -1]

col1, col2 = st.columns(2)

with col1:
    fig_topics = px.bar(
        valid_topics, x='Name', y='Count',
        title='Topic Size Distribution',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig_topics.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-20
    )
    st.plotly_chart(fig_topics, use_container_width=True)

with col2:
    topic_sentiment = df[df['topic_id'] != -1].groupby(['topic_name', 'distilbert_sentiment']).size().reset_index(name='count')
    fig_ts = px.bar(
        topic_sentiment, x='topic_name', y='count', color='distilbert_sentiment',
        title='Sentiment Within Each Topic',
        color_discrete_map={'positive':'#50C878','negative':'#FF6B6B'},
        barmode='group'
    )
    fig_ts.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-20
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# Human-readable topic names
st.markdown("#### Discovered Topics")
topic_labels = {
    '0_therapist_therapy_like_want': '🛋️ Therapy & Treatment',
    '1_avpd_social_disorder_social anxiety': '😰 AvPD & Social Anxiety',
    '2_anxiety_like_feel_ive': '💭 General Anxiety & Feelings',
    '3_dont_like_feel_people': '👥 Loneliness & Relationships'
}

for raw_name, label in topic_labels.items():
    count = len(df[df['topic_name'] == raw_name])
    neg_count = len(df[(df['topic_name'] == raw_name) & (df['distilbert_sentiment'] == 'negative')])
    neg_pct_topic = neg_count / count * 100 if count > 0 else 0
    st.markdown(f"**{label}** — {count:,} posts · {neg_pct_topic:.0f}% negative sentiment")

st.markdown("---")

# ── SECTION 6: WORD CLOUDS ─────────────────────────────────────
st.markdown('<div class="section-title">☁️ Word Clouds</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

def make_wordcloud(text_series, colormap, title):
    text = ' '.join(text_series.fillna('').tolist())
    if len(text) < 10:
        return None
    wc = WordCloud(
        width=600, height=300,
        background_color='#0E1117',
        colormap=colormap,
        max_words=80,
        collocations=False
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=12, pad=10)
    return fig

with col1:
    neg_text = filtered_df[filtered_df['distilbert_sentiment'] == 'negative']['clean_text']
    fig_neg = make_wordcloud(neg_text, 'Reds', 'Negative Posts — Most Common Words')
    if fig_neg:
        st.pyplot(fig_neg)

with col2:
    pos_text = filtered_df[filtered_df['distilbert_sentiment'] == 'positive']['clean_text']
    fig_pos = make_wordcloud(pos_text, 'Greens', 'Positive Posts — Most Common Words')
    if fig_pos:
        st.pyplot(fig_pos)

st.markdown("---")

# ── SECTION 7: COMMUNITY DEEP DIVE ────────────────────────────
st.markdown('<div class="section-title">🏘️ Community Deep Dive</div>', unsafe_allow_html=True)

community_stats = df.groupby('community').agg(
    total_posts=('title', 'count'),
    negative_pct=('distilbert_sentiment', lambda x: (x == 'negative').mean() * 100),
    avg_vader_score=('vader_score', 'mean'),
    dominant_emotion=('emotion', lambda x: x.value_counts().index[0])
).reset_index()

community_stats['negative_pct'] = community_stats['negative_pct'].round(1)
community_stats['avg_vader_score'] = community_stats['avg_vader_score'].round(3)
community_stats.columns = ['Community', 'Total Posts', 'Negative %', 'Avg VADER Score', 'Dominant Emotion']

st.dataframe(community_stats, use_container_width=True)

fig_comm = px.bar(
    community_stats, x='Community', y='Negative %',
    title='Negativity Level by Community',
    color='Negative %',
    color_continuous_scale='Reds',
    text='Negative %'
)
fig_comm.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig_comm, use_container_width=True)

st.markdown("---")

# ── SECTION 8: RAW DATA EXPLORER ──────────────────────────────
st.markdown('<div class="section-title">🔎 Data Explorer</div>', unsafe_allow_html=True)

show_cols = ['title', 'community', 'vader_sentiment', 'lr_sentiment', 'distilbert_sentiment', 'emotion', 'topic_name']
st.dataframe(
    filtered_df[show_cols].head(100).reset_index(drop=True),
    use_container_width=True
)

st.markdown("---")
st.markdown("*Built with PySpark · VADER · Logistic Regression · DistilBERT · BERTopic · Streamlit*")


# ── SECTION 9: PYSPARK BIG DATA RESULTS ───────────────────────
st.markdown('<div class="section-title">⚡ PySpark Big Data Analytics</div>', unsafe_allow_html=True)

import os
spark_metrics_path = "models/spark_metrics.json"

if os.path.exists(spark_metrics_path):
    with open(spark_metrics_path) as f:
        spark_metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spark ML Accuracy", f"{spark_metrics['spark_ml_accuracy']:.2%}")
    col2.metric("Training Samples", f"{spark_metrics['training_samples']:,}")
    col3.metric("Test Samples", f"{spark_metrics['test_samples']:,}")
    col4.metric("TF-IDF Features", f"{spark_metrics['features']:,}")

    st.markdown(f"**Algorithm:** {spark_metrics['algorithm']}")

    spark_community_data = {
        'Community': ['Depression', 'Mindfulness', 'Mental Health', 'Social Anxiety', 'Anxiety'],
        'Negativity %': [92.6, 91.1, 84.2, 80.6, 78.5],
        'Avg VADER Score': [-0.3489, -0.2959, -0.0880, -0.0878, -0.1944],
        'Total Posts': [527, 472, 386, 377, 404]
    }
    spark_df = pd.DataFrame(spark_community_data)

    fig_spark = px.bar(
        spark_df, x='Community', y='Negativity %',
        title='Negativity Rate by Community (Computed via Spark SQL)',
        color='Negativity %',
        color_continuous_scale='OrRd',
        text='Negativity %'
    )
    fig_spark.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_spark.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_spark, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    ⚡ <b>PySpark Pipeline:</b> Data was processed using Apache Spark 4.0 with a distributed 
    TF-IDF + Logistic Regression MLlib pipeline achieving <b>86.68% accuracy</b> — outperforming 
    the scikit-learn baseline (75.93%). Spark SQL was used to compute community-level 
    sentiment aggregations across 4,620 records with 10,000 TF-IDF features.
    </div>
    """, unsafe_allow_html=True)

    # Model accuracy comparison
    model_accuracy_data = pd.DataFrame({
        'Model': ['VADER (Rule-based)', 'Scikit-learn LR', 'Spark MLlib LR', 'DistilBERT'],
        'Accuracy': [None, 0.7593, spark_metrics['spark_ml_accuracy'], None],
        'Type': ['Rule-based', 'Statistical ML', 'Distributed ML', 'Deep Learning']
    })
    model_accuracy_data = model_accuracy_data.dropna()

    fig_acc = px.bar(
        model_accuracy_data, x='Model', y='Accuracy',
        title='Model Accuracy Comparison',
        color='Type',
        text='Accuracy',
        color_discrete_sequence=['#4A90D9', '#50C878', '#FF9F40']
    )
    fig_acc.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_acc.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        yaxis_range=[0, 1.1]
    )
    st.plotly_chart(fig_acc, use_container_width=True)

st.markdown("---")
st.markdown("*Built with PySpark 4.0 · VADER · Scikit-learn · DistilBERT · BERTopic · Streamlit*")


# ── SECTION 10: BIG DATA SCALE ─────────────────────────────────
st.markdown('<div class="section-title">🚀 Big Data Scale — Sentiment140 (1.6M Tweets)</div>', unsafe_allow_html=True)

bigdata_path = "models/bigdata_metrics.json"
if os.path.exists(bigdata_path):
    with open(bigdata_path) as f:
        bd = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records Processed", "1,600,000")
    col2.metric("Training Samples", f"{bd['train_records']:,}")
    col3.metric("Test Samples", f"{bd['test_records']:,}")
    col4.metric("Accuracy", f"{bd['accuracy']:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Time", f"{bd['training_time_seconds']}s")
        st.metric("TF-IDF Features", f"{bd['features']:,}")
        st.markdown(f"**Algorithm:** {bd['algorithm']}")

    with col2:
        scale_data = pd.DataFrame({
            'Dataset': ['Reddit Mental Health', 'Sentiment140'],
            'Records': [4620, 1600000],
            'Type': ['Domain-specific', 'Large-scale']
        })
        fig_scale = px.bar(
            scale_data, x='Dataset', y='Records',
            title='Dataset Scale Comparison',
            color='Type',
            color_discrete_sequence=['#4A90D9', '#FF9F40'],
            text='Records'
        )
        fig_scale.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_scale.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis_range=[0, 2000000]
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    🚀 <b>Big Data Achievement:</b> Spark processed 1,600,000 tweets in 7.4 seconds 
    and trained a TF-IDF + Logistic Regression model on 1,280,209 samples in 29.7 seconds 
    achieving 76.72% accuracy — demonstrating genuine distributed big data processing 
    at scale on commodity hardware.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 11: LIVE STREAM ────────────────────────────────────
st.markdown('<div class="section-title">📡 Real-Time Sentiment Stream — HackerNews Live</div>', unsafe_allow_html=True)

stream_path = "models/stream_summary.json"
live_path = "data/live_stream.csv"

if os.path.exists(stream_path) and os.path.exists(live_path):
    with open(stream_path) as f:
        ss = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts Collected", ss['total_collected'])
    col2.metric("Positive", f"{ss['positive_pct']}%")
    col3.metric("Negative", f"{ss['negative_pct']}%")
    col4.metric("Neutral", f"{ss['neutral_pct']}%")
    st.caption(f"Collected at: {ss['collected_at']} | Source: {ss['source']}")

    live_df = pd.read_csv(live_path)

    col1, col2 = st.columns(2)

    with col1:
        stream_counts = live_df['sentiment'].value_counts().reset_index()
        stream_counts.columns = ['sentiment', 'count']
        fig_stream = px.pie(
            stream_counts, values='count', names='sentiment',
            title='Live Stream Sentiment Distribution',
            color='sentiment',
            color_discrete_map={
                'positive': '#50C878',
                'negative': '#FF6B6B',
                'neutral': '#4A90D9'
            },
            hole=0.4
        )
        fig_stream.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_stream, use_container_width=True)

    with col2:
        # Sentiment over time (rolling)
        live_df['index'] = range(len(live_df))
        live_df['sentiment_score'] = live_df['score'].astype(float)
        live_df['rolling_sentiment'] = live_df['sentiment_score'].rolling(window=10).mean()

        fig_roll = px.line(
            live_df, x='index', y='rolling_sentiment',
            title='Sentiment Trend (Rolling Average)',
            color_discrete_sequence=['#4A90D9']
        )
        fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_roll.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title='Post Number',
            yaxis_title='Sentiment Score'
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    # Live posts table
    st.markdown("#### Sample Live Posts")
    display_live = live_df[['title', 'sentiment', 'score']].head(20).reset_index(drop=True)
    display_live.columns = ['Headline', 'Sentiment', 'Score']
    st.dataframe(display_live, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📡 <b>Real-Time Monitoring:</b> 200 live HackerNews posts were streamed and scored 
    in real-time using VADER. Results show 56.5% neutral, 23.5% positive, 20.0% negative 
    sentiment — consistent with tech community discourse patterns where most posts are 
    informational rather than emotionally charged.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Built with PySpark 4.0 · VADER · Scikit-learn · DistilBERT · BERTopic · HackerNews API · Streamlit*")
