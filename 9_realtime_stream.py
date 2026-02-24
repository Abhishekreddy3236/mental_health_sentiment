import json, time, csv, os, re, requests
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_FILE = "data/live_stream.csv"
MAX_POSTS = 200
analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05: return 'positive', score
    elif score <= -0.05: return 'negative', score
    else: return 'neutral', score

def fetch_ids():
    all_ids = []
    for feed in ['newstories', 'topstories', 'beststories']:
        try:
            r = requests.get(f"https://hacker-news.firebaseio.com/v0/{feed}.json", timeout=5)
            all_ids.extend(r.json()[:200])
        except:
            pass
    return list(set(all_ids))

def get_item(item_id):
    try:
        r = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=5)
        return r.json()
    except:
        return None

def main():
    print("=" * 55)
    print("HACKERNEWS REAL-TIME SENTIMENT STREAM")
    print("=" * 55)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['timestamp', 'title', 'sentiment', 'score'])

    posts_collected = []
    stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
    seen_ids = set()

    print(f"Target: {MAX_POSTS} posts — fetching from new/top/best feeds\n")

    while len(posts_collected) < MAX_POSTS:
        try:
            story_ids = fetch_ids()
            new_ids = [i for i in story_ids if i not in seen_ids]
            print(f"  Fetched {len(new_ids)} new story IDs...")

            for story_id in new_ids:
                seen_ids.add(story_id)
                item = get_item(story_id)
                if not item or item.get('type') != 'story':
                    continue
                title = item.get('title', '')
                if not title or len(title) < 10:
                    continue
                clean = clean_text(title)
                if len(clean) < 10:
                    continue

                sentiment, score = get_sentiment(clean)
                stats[sentiment] += 1
                timestamp = datetime.fromtimestamp(item.get('time', time.time()))

                posts_collected.append({'title': title, 'sentiment': sentiment, 'score': score})

                with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([str(timestamp), title, sentiment, round(score, 4)])

                print(f"[{len(posts_collected):03d}] {emoji[sentiment]} {sentiment.upper():<8} | {title[:65]}")

                if len(posts_collected) >= MAX_POSTS:
                    break
                time.sleep(0.05)

            if len(posts_collected) < MAX_POSTS:
                remaining = MAX_POSTS - len(posts_collected)
                print(f"\n  Need {remaining} more posts. Waiting 15s for new stories...\n")
                seen_ids.clear()  # reset so we can refetch
                time.sleep(15)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(5)

    total = len(posts_collected)
    print("\n" + "=" * 55)
    print("LIVE STREAM SUMMARY")
    print("=" * 55)
    print(f"Total posts collected: {total}")
    if total > 0:
        print(f"🟢 Positive: {stats['positive']} ({stats['positive']/total*100:.1f}%)")
        print(f"🔴 Negative: {stats['negative']} ({stats['negative']/total*100:.1f}%)")
        print(f"🟡 Neutral:  {stats['neutral']} ({stats['neutral']/total*100:.1f}%)")

        summary = {
            'total_collected': total,
            'source': 'HackerNews Live API',
            'positive': stats['positive'],
            'negative': stats['negative'],
            'neutral': stats['neutral'],
            'positive_pct': round(stats['positive']/total*100, 1),
            'negative_pct': round(stats['negative']/total*100, 1),
            'neutral_pct': round(stats['neutral']/total*100, 1),
            'collected_at': str(datetime.now())
        }
        os.makedirs("models", exist_ok=True)
        with open("models/stream_summary.json", "w") as f:
            json.dump(summary, f)
        print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
