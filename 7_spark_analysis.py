import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, when, length, explode, split, lower, regexp_replace, round as spark_round
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json, os

def create_spark_session():
    spark = SparkSession.builder \
        .appName("MentalHealthSentimentAnalysis") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark session created | Version: {spark.version}")
    return spark

def load_and_clean(spark):
    print("\n[1] Loading data into Spark...")
    df = spark.read.csv("data/final_results.csv", header=True, inferSchema=False)

    # Cast numeric columns explicitly
    df = df.withColumn("vader_score", col("vader_score").cast(DoubleType()))
    df = df.withColumn("lr_confidence", col("lr_confidence").cast(DoubleType()))
    df = df.withColumn("distilbert_score", col("distilbert_score").cast(DoubleType()))

    # Keep only valid community rows
    valid_communities = ["Depression", "Anxiety", "Mental Health", "Social Anxiety", "Mindfulness"]
    df = df.filter(col("community").isin(valid_communities))
    df = df.dropDuplicates(["title"])

    print(f"Total records after cleaning: {df.count()}")
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    return df

def spark_sql_analytics(spark, df):
    print("\n[2] Running Spark SQL Analytics...")
    df.createOrReplaceTempView("posts")

    print("\nSentiment by community:")
    spark.sql("""
        SELECT community,
               distilbert_sentiment,
               COUNT(*) as post_count
        FROM posts
        WHERE distilbert_sentiment IS NOT NULL
        GROUP BY community, distilbert_sentiment
        ORDER BY community, post_count DESC
    """).show(20)

    print("\nNegativity rate by community:")
    spark.sql("""
        SELECT community,
               COUNT(*) as total,
               SUM(CASE WHEN distilbert_sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
               ROUND(SUM(CASE WHEN distilbert_sentiment = 'negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as negative_pct,
               ROUND(AVG(vader_score), 4) as avg_vader_score
        FROM posts
        WHERE community IS NOT NULL
        GROUP BY community
        ORDER BY negative_pct DESC
    """).show()

    print("\nEmotion distribution:")
    spark.sql("""
        SELECT emotion, COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM posts
        WHERE emotion IS NOT NULL
        GROUP BY emotion
        ORDER BY count DESC
    """).show()

def spark_ml_pipeline(spark, df):
    print("\n[3] Running Spark ML Pipeline (TF-IDF + Logistic Regression)...")

    df = df.withColumn("label",
        when(col("distilbert_sentiment") == "positive", 1.0).otherwise(0.0)
    ).filter(col("clean_text").isNotNull())

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=20, regParam=0.01)
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count()} | Test: {test_df.count()}")

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"\n  Spark ML Accuracy: {accuracy:.4f}")

    metrics = {
        "spark_ml_accuracy": round(accuracy, 4),
        "training_samples": train_df.count(),
        "test_samples": test_df.count(),
        "algorithm": "TF-IDF + Logistic Regression (Spark MLlib)",
        "features": 10000
    }
    os.makedirs("models", exist_ok=True)
    with open("models/spark_metrics.json", "w") as f:
        json.dump(metrics, f)

    print("  Saved to models/spark_metrics.json")
    return metrics

def word_frequency(spark, df):
    print("\n[4] Word Frequency Analysis...")

    words_df = df.select(
        explode(split(lower(regexp_replace(col("clean_text"), "[^a-zA-Z\\s]", "")), "\\s+")).alias("word"),
        col("distilbert_sentiment")
    ).filter(col("word").rlike("^[a-z]{4,}$"))

    stop = ["that", "this", "with", "have", "from", "they", "will",
            "been", "were", "when", "what", "just", "like", "feel",
            "really", "about", "some", "would", "know", "dont"]
    words_df = words_df.filter(~col("word").isin(stop))

    print("\nTop words — NEGATIVE posts:")
    words_df.filter(col("distilbert_sentiment") == "negative") \
        .groupBy("word").count().orderBy(col("count").desc()).show(12)

    print("\nTop words — POSITIVE posts:")
    words_df.filter(col("distilbert_sentiment") == "positive") \
        .groupBy("word").count().orderBy(col("count").desc()).show(12)

def main():
    print("=" * 55)
    print("PYSPARK BIG DATA ANALYTICS PIPELINE")
    print("=" * 55)

    spark = create_spark_session()
    df = load_and_clean(spark)
    spark_sql_analytics(spark, df)
    metrics = spark_ml_pipeline(spark, df)
    word_frequency(spark, df)

    print("\n" + "=" * 55)
    print(f"Done! Spark ML Accuracy: {metrics['spark_ml_accuracy']:.2%}")
    spark.stop()

if __name__ == "__main__":
    main()
