from newspaper import Article
import requests
from sentence_transformers import SentenceTransformer
import chromadb
import time

NEWS_SOURCES = [
    "https://rss.cnn.com/rss/cnn_topstories.rss",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml"
]


model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("news")

import feedparser

def fetch_rss_articles():
    articles = []
    for feed_url in NEWS_SOURCES:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            url = entry.link
            try:
                article = Article(url)
                article.download()
                article.parse()
                articles.append({
                    "title": article.title,
                    "text": article.text,
                    "url": url,
                    "source": feed.feed.title if hasattr(feed.feed, 'title') else "Unknown"
                })
            except Exception:
                continue  # Skip articles that fail to download/parse
    return articles

from fact_checking.misinfo import detect_misinformation

def ingest_news(max_articles=5, progress_callback=None):
    articles = fetch_rss_articles()
    count = 0
    for idx, art in enumerate(articles[:max_articles]):
        embedding = model.encode(art['text'])
        misinfo_verdict, misinfo_explanation = detect_misinformation(art['text'])
        collection.add(
            ids=[art['url']],
            documents=[art['text']],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "title": art['title'],
                "url": art['url'],
                "source": art['source'],
                "misinfo_verdict": misinfo_verdict,
                "misinfo_explanation": misinfo_explanation
            }]
        )
        count += 1
        if progress_callback:
            progress_callback(idx + 1, min(max_articles, len(articles)))
    return count

if __name__ == "__main__":
    articles = fetch_rss_articles()
    print(f"Fetched {len(articles)} articles.")
    for a in articles[:2]:
        print(a['title'], a['url'])
