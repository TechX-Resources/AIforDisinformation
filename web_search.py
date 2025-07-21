import requests
from bs4 import BeautifulSoup
import wikipediaapi

# Replace with your API keys
GOOGLE_FACT_CHECK_API_KEY = "YOUR_GOOGLE_API_KEY"
GNEWS_API_KEY = "YOUR_GNEWS_API_KEY"

def verify_with_google_fact_check(claim):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={GOOGLE_FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        claims = data.get("claims", [])
        return [c["text"] + " - " + c["claimReview"][0]["textualRating"] for c in claims] if claims else ["No result."]
    return ["Google Fact Check API error."]

def verify_with_gnews(claim):
    url = f"https://gnews.io/api/v4/search?q={claim}&token={GNEWS_API_KEY}&lang=en"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [f"{a['title']} - {a['source']['name']}" for a in articles[:3]]
    return ["GNews API error."]

def verify_with_wikipedia(claim):
    wiki = wikipediaapi.Wikipedia(
        user_agent="FactCheckBot/1.0 (contact: youremail@example.com)",
        language="en"
    )
    page = wiki.page(claim)
    return [page.summary[:500]] if page.exists() else ["No Wikipedia match."]


def verify_with_snopes(claim):
    search_url = f"https://www.snopes.com/?s={claim.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.select("article h2.entry-title a")
    return [link.text.strip() for link in results[:3]] if results else ["No Snopes result."]

def verify_claim(claim):
    print(f"\n🔎 Verifying Claim: \"{claim}\"\n")

    sources = {
        "Google Fact Check": verify_with_google_fact_check(claim),
        "Wikipedia Summary": verify_with_wikipedia(claim),
        "GNews Articles": verify_with_gnews(claim),
        "Snopes Results": verify_with_snopes(claim)
    }

    for source, results in sources.items():
        print(f"{source}:")
        for r in results:
            print("  •", r)
        print()
    return sources

from duckduckgo_search import DDGS

def verify_with_duckduckgo(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", "")
            results.append(f"{title}: {snippet} (Source: {url})")
    return results
