from crewai import Task
from agents.extractor_agent import extract_text_from_pdf_tool, transcribe_audio_tool, upsert_to_qdrant_tool
from agents.rag_agent import query_rag  # tool object
from agents.answer_agent import call_llm  # tool object
from agents.improver_agent import improve_answer  # tool object
from agents.search_fallback_agent import web_search  # tool object
from core.embeddings import get_embeddings

class RAGQueryTask(Task):
    def run(self, query: str, collection: str):

        # Step 1: Query Qdrant
        search_results = query_rag.run(collection=collection, query=query)
        context = "\n\n---\n\n".join([d["payload"]["text"] for d in search_results])

        # Step 2: Draft answer
        draft_prompt = (
            f"You are an assistant. Use the context to answer:\n\n{context}\n\n"
            f"Question: {query}"
        )
        draft_answer = call_llm.run(draft_prompt)

        # Step 3: Evaluate confidence
        confidence = max([d.get("score", 0.0) for d in search_results], default=0.0)

        # ---------------------------------------------------------
        # CASE A — GOOD QDRANT SCORE ? RETURN DRAFT ANSWER
        # ---------------------------------------------------------
        if confidence >= 0.8:
            return {
                "answer": draft_answer,
                "sources": [d["payload"].get("source", "unknown") for d in search_results],
                "confidence": confidence,
            }

        # ---------------------------------------------------------
        # CASE B — LOW SCORE ? TRY IMPROVE ANSWER
        # ---------------------------------------------------------
        improved = improve_answer.run(query=query, docs=context, draft=draft_answer)

        # If improvement succeeded (not insufficient)
        if "INSUFFICIENT" not in improved:
            return {
                "answer": improved,
                "sources": [d["payload"].get("source", "unknown") for d in search_results],
                "confidence": confidence,
            }

        # ---------------------------------------------------------
        # CASE C — IMPROVEMENT FAILED ? FALLBACK WEB SEARCH
        # ---------------------------------------------------------
        web_results = web_search.run(query=query)

        if web_results:
            snippets = [r["snippet"] for r in web_results]
            metas = [{"source": r.get("link", "unknown")} for r in web_results]
            embeddings = get_embeddings(snippets)

            # Insert fallback search results into RAG
            upsert_to_qdrant_tool.run(collection, snippets, metas, embeddings)

            # Re-query with new knowledge
            search_results = query_rag.run(collection=collection, query=query)
            context = "\n\n---\n\n".join([d["payload"]["text"] for d in search_results])

            final_prompt = (
                f"You are an assistant. Use the context to answer:\n\n{context}\n\n"
                f"Question: {query}"
            )
            final_answer = call_llm.run(final_prompt)

            confidence = max([d.get("score", 0.0) for d in search_results], default=0.0)

            return {
                "answer": final_answer,
                "sources": [d["payload"].get("source", "unknown") for d in search_results],
                "confidence": confidence,
            }

        # ---------------------------------------------------------
        # CASE D — NO WEB RESULTS ? RETURN IMPROVED OR DRAFT
        # ---------------------------------------------------------
        return {
            "answer": improved if "INSUFFICIENT" not in improved else draft_answer,
            "sources": [d["payload"].get("source", "unknown") for d in search_results],
            "confidence": confidence,
        }
