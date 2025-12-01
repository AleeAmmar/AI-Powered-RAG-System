# agents/evaluator_agent.py
from crewai import Agent
from agents.answer_agent import call_llm  # LLM tool to judge answers

# ---------------------------
# Evaluator Agent
# ---------------------------
evaluator_agent = Agent(
    name="Evaluator",
    role="Answer Evaluator",
    goal="Assess the quality and confidence of generated answers based on retrieved context and LLM judgment.",
    backstory="You are an expert at evaluating LLM answers and retrieval results, providing a confidence score for correctness and grounding.",
    description="Evaluate answer confidence using retrieval scores and optional LLM judgment.",
    tools=[]  # no external tools needed besides call_llm which is used in task function
)

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_task_fn(query: str, answer: str, search_results: list, context: str):
    """
    Computes confidence of the answer.
    Combines max retrieval score with LLM-based judgment.
    Returns float between 0 and 1.
    """
    # Step 1: Max retrieval score
    r_conf = max((d.get("score", 0.0) for d in search_results), default=0.0)

    # Step 2: LLM judge
    llm_prompt = (
        f"Rate the following answer 0-1 for correctness and grounding given the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:\n{answer}\nScore (0-1):"
    )

    try:
        llm_score = float(call_llm.run(llm_prompt).strip().split()[0])
    except Exception:
        llm_score = 0.0

    # Step 3: Return the maximum of retrieval score and LLM judgment
    return max(r_conf, llm_score)
