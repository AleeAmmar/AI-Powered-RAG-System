from crewai import Agent
from crewai.tools import tool
import requests
from config import LM_STUDIO_URL, LM_MODEL

# LM Studio tool
@tool("call_llm")
def call_llm(prompt: str) -> str:
    """Call LM Studio LLM with a prompt and return the response."""
    payload = {
    "model": LM_MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 500,
    "temperature": 0.0
}
    resp = requests.post(LM_STUDIO_URL + 'chat/completions', json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# Create the Answer Agent
answer_agent = Agent(
    name="AnswerAgent",
    role="answer_generator",
    goal="Generate high-quality answers to user queries using LM Studio",
    backstory="This agent receives prompts and returns answers using a local LM Studio model.",
    tools=[call_llm]
)