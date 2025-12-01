from crewai import Crew, Task
from pydantic import Field

# Import your existing agents and tools
from agents.rag_agent import rag_agent, query_rag
from agents.answer_agent import answer_agent, call_llm
from agents.improver_agent import improver_agent, improve_answer
from agents.search_fallback_agent import search_fallback_agent, web_search
from agents.extractor_agent import extractor_agent, upsert_to_qdrant_tool
from agents.evaluator_agent import evaluator_agent
from core.embeddings import get_embeddings

# ---------------------------
# Task functions
# ---------------------------
def retrieve_task_fn(query, collection):
    results = query_rag.run(collection=collection, query=query)
    context = "\n\n---\n\n".join([d["payload"]["text"] for d in results])
    return {"search_results": results, "context": context}

def draft_task_fn(query, context):
    prompt = f"You are an assistant. Use the context to answer:\n\n{context}\n\nQuestion: {query}"
    return call_llm.run(prompt)

def improve_task_fn(query, context, draft):
    improved = improve_answer.run(query=query, docs=context, draft=draft)
    return improved if "INSUFFICIENT" not in improved else None

def webfallback_task_fn(query, collection):
    web_results = web_search.run(query=query)
    if not web_results:
        return None
    snippets = [r["snippet"] for r in web_results]
    metas = [{"source": r.get("link", "unknown")} for r in web_results]
    embeddings = get_embeddings(snippets)
    upsert_to_qdrant_tool.run(collection, snippets, metas, embeddings)
    return True

def evaluate_task_fn(query, answer, search_results, context):
    r_conf = max((d.get("score", 0.0) for d in search_results), default=0.0)
    llm_prompt = (
        f"Rate the following answer 0-1 for correctness and grounding given the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:\n{answer}\nScore (0-1):"
    )
    try:
        llm_score = float(call_llm.run(llm_prompt).strip().split()[0])
    except Exception:
        llm_score = 0.0
    return max(r_conf, llm_score)

# ---------------------------
# Task objects (metadata for Crew)
# ---------------------------
retrieve_task = Task(
    name="RetrieveTask",
    description="Retrieve documents from Qdrant",
    agent=rag_agent,
    expected_output="Dictionary with 'search_results' and 'context'"
)

draft_task = Task(
    name="DraftTask",
    description="Generate draft answer",
    agent=answer_agent,
    expected_output="Draft answer as a string"
)

improve_task = Task(
    name="ImproveTask",
    description="Improve draft answer if confidence low",
    agent=improver_agent,
    expected_output="Improved answer string or None"
)

webfallback_task = Task(
    name="WebFallbackTask",
    description="Perform web search fallback and insert new knowledge",
    agent=search_fallback_agent,
    expected_output="Boolean indicating if new knowledge was inserted"
)

evaluate_task = Task(
    name="EvaluateTask",
    description="Evaluate final answer confidence",
    agent=evaluator_agent,
    expected_output="Float confidence score between 0 and 1"
)

# ---------------------------
# Conditional RAG Crew
# ---------------------------
class ConditionalRAGCrew(Crew):
    collection: str = Field(..., description="Qdrant collection to query")

    def retrieve_task_fn(self, query):
        return retrieve_task_fn(query, self.collection)

    def draft_task_fn(self, query, context):
        return draft_task_fn(query, context)

    def improve_task_fn(self, query, context, draft):
        return improve_task_fn(query, context, draft)

    def webfallback_task_fn(self, query):
        return webfallback_task_fn(query, self.collection)

    def evaluate_task_fn(self, query, answer, search_results, context):
        return evaluate_task_fn(query, answer, search_results, context)

    def kickoff(self, query: str):
        # -------------------------
        # Stage 1: Retrieve
        # -------------------------
        retrieve_out = retrieve_task_fn(query=query, collection=self.collection)
        context = retrieve_out["context"]
        search_results = retrieve_out["search_results"]
        r_conf = max((d.get("score", 0.0) for d in search_results), default=0.0)

        # -------------------------
        # Stage 2: Draft
        # -------------------------
        draft_answer = draft_task_fn(query=query, context=context)

        # -------------------------
        # Stage 3: Conditional Improve
        # -------------------------
        if r_conf < 0.8:
            improved = improve_task_fn(query=query, context=context, draft=draft_answer)
            if improved:
                draft_answer = improved
            else:
                fallback_done = webfallback_task_fn(query=query, collection=self.collection)
                if fallback_done:
                    # Re-run retrieval + draft
                    retrieve_out = retrieve_task_fn(query=query, collection=self.collection)
                    context = retrieve_out["context"]
                    search_results = retrieve_out["search_results"]
                    draft_answer = draft_task_fn(query=query, context=context)

        # -------------------------
        # Stage 4: Evaluate
        # -------------------------
        confidence = evaluate_task_fn(query=query, answer=draft_answer, search_results=search_results, context=context)

        return {
            "answer": draft_answer,
            "sources": [d["payload"].get("source", "unknown") for d in search_results],
            "confidence": confidence
        }
