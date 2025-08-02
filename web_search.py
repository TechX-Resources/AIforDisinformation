from ddgs import DDGS


def verify_with_duckduckgo(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", "")
            results.append(f"{title}: {snippet} (Source: {url})")
    return results
