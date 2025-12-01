from crewai import Agent
from crewai.tools import tool
import os
import requests
from config import SERPAPI_API_KEY


@tool("web_search")
def web_search(query: str, num: int = 5):
    """
    Perform live web search using SerpAPI and return top results.
    Returns a list of dicts: [{'title':..., 'snippet':..., 'link':...}]
    """
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY environment variable is not set.")

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num
    }

    resp = requests.get("https://serpapi.com/search", params=params)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })

    return results

# Create the Search Fallback Agent
search_fallback_agent = Agent(
    name="SearchFallback",
    role="fallback_searcher",
    goal="Perform live web search if no RAG results are found",
    backstory="This agent queries the web using SerpAPI to provide additional context when RAG fails.",
    tools=[web_search]
)

