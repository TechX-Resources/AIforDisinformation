import requests
from bs4 import BeautifulSoup
import wikipediaapi
import logging
import os

from duckduckgo_search import DDGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your API keys
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

if not GOOGLE_FACT_CHECK_API_KEY or not GNEWS_API_KEY:
    logger.error("API keys not set. Please set GOOGLE_FACT_CHECK_API_KEY and GNEWS_API_KEY as environment variables.")
    raise EnvironmentError("API keys not set.")


def verify_with_google_fact_check(claim):
    """
    Query Google Fact Check API for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    try:
        url = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            f"?query={claim}&key={GOOGLE_FACT_CHECK_API_KEY}"
        )
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            for c in claims:
                results.append({
                    "source": "Google Fact Check",
                    "title": c.get("text", ""),
                    "summary": c.get("claimReview", [{}])[0].get("textualRating", ""),
                    "url": c.get("claimReview", [{}])[0].get("url", "")
                })
            if not results:
                results.append({
                    "source": "Google Fact Check",
                    "title": "",
                    "summary": "No result.",
                    "url": ""
                })
            return results
        logger.error("Google Fact Check API error: %s", response.status_code)
        results.append({
            "source": "Google Fact Check",
            "title": "",
            "summary": "Google Fact Check API error.",
            "url": ""
        })
        return results
    except Exception as e:
        logger.exception("Exception in verify_with_google_fact_check")
        return [{
            "source": "Google Fact Check",
            "title": "",
            "summary": f"Error: {str(e)}",
            "url": ""
        }]


def verify_with_gnews(claim):
    """
    Query GNews API for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    try:
        url = (
            f"https://gnews.io/api/v4/search?q={claim}&token={GNEWS_API_KEY}&lang=en"
        )
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            for a in articles[:3]:
                results.append({
                    "source": "GNews",
                    "title": a.get("title", ""),
                    "summary": "",
                    "url": a.get("url", "")
                })
            if not results:
                results.append({
                    "source": "GNews",
                    "title": "",
                    "summary": "No result.",
                    "url": ""
                })
            return results
        logger.error("GNews API error: %s", response.status_code)
        results.append({
            "source": "GNews",
            "title": "",
            "summary": "GNews API error.",
            "url": ""
        })
        return results
    except Exception as e:
        logger.exception("Exception in verify_with_gnews")
        return [{
            "source": "GNews",
            "title": "",
            "summary": f"Error: {str(e)}",
            "url": ""
        }]


def verify_with_wikipedia(claim):
    """
    Query Wikipedia for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent="FactCheckBot/1.0 (contact: youremail@example.com)",
            language="en"
        )
        page = wiki.page(claim)
        if page.exists():
            return [{
                "source": "Wikipedia",
                "title": page.title,
                "summary": page.summary[:500],
                "url": page.fullurl
            }]
        else:
            return [{
                "source": "Wikipedia",
                "title": claim,
                "summary": "No Wikipedia match.",
                "url": ""
            }]
    except Exception as e:
        logger.exception("Exception in verify_with_wikipedia")
        return [{
            "source": "Wikipedia",
            "title": claim,
            "summary": f"Error: {str(e)}",
            "url": ""
        }]


def verify_with_snopes(claim):
    """
    Scrape Snopes for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    try:
        search_url = f"https://www.snopes.com/?s={claim.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select("article h2.entry-title a")
        if results:
            return [{
                "source": "Snopes",
                "title": link.text.strip(),
                "summary": "",
                "url": link.get("href", "")
            } for link in results[:3]]
        else:
            return [{
                "source": "Snopes",
                "title": "",
                "summary": "No Snopes result.",
                "url": ""
            }]
    except Exception as e:
        logger.exception("Exception in verify_with_snopes")
        return [{
            "source": "Snopes",
            "title": "",
            "summary": f"Error: {str(e)}",
            "url": ""
        }]


def verify_with_duckduckgo(query, max_results=5):
    """
    Query DuckDuckGo for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "source": "DuckDuckGo",
                    "title": r.get("title", ""),
                    "summary": r.get("body", ""),
                    "url": r.get("href", "")
                })
        if not results:
            results.append({
                "source": "DuckDuckGo",
                "title": "",
                "summary": "No DuckDuckGo result.",
                "url": ""
            })
        return results
    except Exception as e:
        logger.exception("Exception in verify_with_duckduckgo")
        return [{
            "source": "DuckDuckGo",
            "title": "",
            "summary": f"Error: {str(e)}",
            "url": ""
        }]


def verify_claim(claim):
    """
    Aggregate results from all sources for a claim.
    Returns a list of dicts with keys: source, title, summary, url.
    """
    logger.info("🔎 Verifying Claim: \"%s\"", claim)
    sources = (
        verify_with_google_fact_check(claim)
        + verify_with_wikipedia(claim)
        + verify_with_gnews(claim)
        + verify_with_snopes(claim)
        + verify_with_duckduckgo(claim)
    )
    for result in sources:
        logger.info("%s:", result['source'])
        logger.info("  • Title: %s", result['title'])
        logger.info("  • Summary: %s", result['summary'])
        logger.info("  • URL: %s\n", result['url'])
    return sources


if __name__ == "__main__":
    # Example usage for contributors
    claim = "Barack Obama"
    print("Testing with claim:", claim)
    results = verify_claim(claim)
    for res in results:
        print(f"{res['source']}: {res['title']} - {res['summary']} ({res['url']})")
