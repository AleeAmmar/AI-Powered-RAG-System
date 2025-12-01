from crewai import Agent
from crewai.tools import tool
from core.qdrant_utils import qdrant_query
from core.embeddings import get_embeddings

# Tool for querying RAG via Qdrant
@tool("qdrant_query")
def query_rag(collection: str, query: str, topk: int = 5):
    """Queries the Qdrant vector database and returns top results."""
    #query_emb = get_embeddings([query])[0]
    results = qdrant_query(query, topk=topk)
    return results

# Create the RAG Agent
rag_agent = Agent(
    name="RAGAgent",
    role="rag_retriever",
    goal="Retrieve relevant context documents from Qdrant for user queries",
    backstory="This agent handles retrieval of vector embeddings and returns the most relevant chunks for RAG.",
    tools=[query_rag]
)
