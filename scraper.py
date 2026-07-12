"""
Multi-source web scraper for African news, data, and documents.
Fetches content from specified URLs and APIs for LLM training data.
"""
import os
import json
import time
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import SCRAPE_SOURCES, NEWS_API_KEY, GUARDIAN_API_KEY, SCRAPER_API_KEY, DATA_DIR


class WebScraper:
    """
    Scrapes multiple web sources for African-focused content.
    Handles news, documents, and data portals.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        self.scraped_dir = os.path.join(DATA_DIR, "scraped_content")
        os.makedirs(self.scraped_dir, exist_ok=True)
        self.rate_limit_delay = 1.0  # seconds between requests

    def fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch content from a URL with error handling."""
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return resp.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def scrape_newsapi(self, query: str = "Africa", page_size: int = 100) -> List[Dict]:
        """Fetch African news from NewsAPI."""
        if not NEWS_API_KEY:
            print("NewsAPI key not set")
            return []

        articles = []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": min(page_size, 100),
            "language": "en",
            "sortBy": "publishedAt",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data.get("status") == "ok":
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "published_at": article.get("publishedAt", ""),
                        "topic": "news",
                    })
                print(f"Fetched {len(articles)} articles from NewsAPI")
            else:
                print(f"NewsAPI error: {data.get('message', 'Unknown')}")
        except Exception as e:
            print(f"NewsAPI fetch error: {e}")

        return articles

    def scrape_guardian(self, query: str = "Africa", page_size: int = 50) -> List[Dict]:
        """Fetch African news from The Guardian."""
        if not GUARDIAN_API_KEY:
            print("Guardian API key not set")
            return []

        articles = []
        url = "https://content.guardianapis.com/search"
        params = {
            "q": query,
            "api-key": GUARDIAN_API_KEY,
            "page-size": min(page_size, 50),
            "show-fields": "bodyText,headline",
            "order-by": "newest",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            results = data.get("response", {}).get("results", [])
            for article in results:
                fields = article.get("fields", {})
                articles.append({
                    "title": fields.get("headline", article.get("webTitle", "")),
                    "content": fields.get("bodyText", ""),
                    "url": article.get("webUrl", ""),
                    "source": "The Guardian",
                    "published_at": article.get("webPublicationDate", ""),
                    "topic": "news",
                })
            print(f"Fetched {len(articles)} articles from The Guardian")
        except Exception as e:
            print(f"Guardian fetch error: {e}")

        return articles

    def scrape_generic_page(self, url: str) -> Optional[Dict]:
        """Scrape a generic web page for text content."""
        html = self.fetch_url(url)
        if not html:
            return None

        text = self.parse_html(html)
        if not text or len(text) < 100:
            return None

        return {
            "url": url,
            "content": text[:50000],  # Limit to 50k chars
            "source": urlparse(url).netloc,
            "topic": "web_scrape",
            "published_at": datetime.now().isoformat(),
        }

    def scrape_all_sources(self, max_pages: int = 20) -> List[Dict]:
        """Scrape content from all configured sources."""
        all_content = []

        # 1. NewsAPI
        newsapi_content = self.scrape_newsapi()
        all_content.extend(newsapi_content)

        # 2. The Guardian
        guardian_content = self.scrape_guardian()
        all_content.extend(guardian_content)

        # 3. Generic web pages from source list
        sources_to_scrape = [s for s in SCRAPE_SOURCES[:max_pages] if not s.startswith("http://www.news.io")]
        for url in tqdm(sources_to_scrape, desc="Scraping web pages"):
            content = self.scrape_generic_page(url)
            if content:
                all_content.append(content)
                print(f"  Scraped: {url[:60]}... ({len(content['content'])} chars)")

        return all_content

    def save_scraped_content(self, content_list: List[Dict]):
        """Save scraped content to files."""
        # Save as JSON
        json_path = os.path.join(self.scraped_dir, "scraped_content.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(content_list, f, indent=2)
        print(f"Saved {len(content_list)} items to {json_path}")

        # Save individual text files
        for i, item in enumerate(content_list):
            content = item.get("content") or item.get("description") or ""
            if content:
                fname = f"scraped_{i:04d}.txt"
                fpath = os.path.join(self.scraped_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(f"Title: {item.get('title', '')}\n")
                    f.write(f"Source: {item.get('source', '')}\n")
                    f.write(f"URL: {item.get('url', '')}\n")
                    f.write(f"Topic: {item.get('topic', '')}\n")
                    f.write("-" * 80 + "\n")
                    f.write(content)

        print(f"Saved {len(content_list)} individual text files to {self.scraped_dir}")


if __name__ == "__main__":
    scraper = WebScraper()
    content = scraper.scrape_all_sources(max_pages=5)
    scraper.save_scraped_content(content)
    print(f"\nTotal scraped items: {len(content)}")