import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, round as spark_round, udf, lower, regexp_replace
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import json
import os

def create_spark_session():
    spark = SparkSession.builder \
        .appName("BigDataSentiment140") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark version: {spark.version}")
    return spark

def load_sentiment140(spark):
    print("\n[1] Loading Sentiment140 (1.6M tweets) into Spark...")
    start = time.time()

    df = spark.read.csv(
        "data/sentiment140.csv",
        header=False,
        inferSchema=False
    ).toDF("polarity", "id", "date", "query", "user", "text")

    # Cast polarity and map to labels
    df = df.withColumn("polarity", col("polarity").cast("integer"))
    df = df.withColumn("sentiment",
        when(col("polarity") == 4, "positive")
        .when(col("polarity") == 0, "negative")
        .otherwise("neutral")
    )
    df = df.withColumn("label",
        when(col("polarity") == 4, 1.0).otherwise(0.0)
    )

    # Clean text
    df = df.withColumn("clean_text",
        lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
    )
    df = df.filter(col("clean_text").isNotNull())
    df = df.filter(col("clean_text") != "")

    total = df.count()
    elapsed = time.time() - start

    print(f"  Records loaded: {total:,}")
    print(f"  Partitions: {df.rdd.getNumPartitions()}")
    print(f"  Load time: {elapsed:.1f}s")
    return df, total

def run_spark_sql_analytics(spark, df):
    print("\n[2] Running Spark SQL on 1.6M records...")
    df.createOrReplaceTempView("tweets")

    print("\nSentiment distribution:")
    spark.sql("""
        SELECT sentiment,
               COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM tweets
        GROUP BY sentiment
        ORDER BY count DESC
    """).show()

    print("\nTweet volume by hour of day:")
    spark.sql("""
        SELECT SUBSTRING(date, 12, 2) as hour,
               COUNT(*) as tweet_count,
               SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,
               SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative
        FROM tweets
        WHERE date IS NOT NULL AND LENGTH(date) > 12
        GROUP BY hour
        ORDER BY hour
    """).show(24)

def run_bigdata_ml(spark, df):
    print("\n[3] Running Spark MLlib on 1.6M records...")

    # Sample for faster training — still 800K rows
    train_full, test_full = df.randomSplit([0.8, 0.2], seed=42)

    print(f"  Training samples: {train_full.count():,}")
    print(f"  Test samples: {test_full.count():,}")

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=50000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.01)

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    print("  Training pipeline (this takes a few minutes on 1.6M rows)...")
    start = time.time()
    model = pipeline.fit(train_full)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    predictions = model.transform(test_full)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"\n  Accuracy on {test_full.count():,} test records: {accuracy:.4f}")

    metrics = {
        "dataset": "Sentiment140",
        "total_records": 1600000,
        "train_records": train_full.count(),
        "test_records": test_full.count(),
        "accuracy": round(accuracy, 4),
        "training_time_seconds": round(train_time, 1),
        "features": 50000,
        "algorithm": "TF-IDF (50K features) + LR — Spark MLlib"
    }

    os.makedirs("models", exist_ok=True)
    with open("models/bigdata_metrics.json", "w") as f:
        json.dump(metrics, f)

    print("  Saved to models/bigdata_metrics.json")
    return metrics

def main():
    print("=" * 55)
    print("BIG DATA PIPELINE — SENTIMENT140 (1.6M TWEETS)")
    print("=" * 55)

    spark = create_spark_session()
    df, total = load_sentiment140(spark)
    run_spark_sql_analytics(spark, df)
    metrics = run_bigdata_ml(spark, df)

    print("\n" + "=" * 55)
    print(f"Done!")
    print(f"Records processed: {total:,}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Training time: {metrics['training_time_seconds']}s")
    spark.stop()

if __name__ == "__main__":
    main()
