import os
import sys
import traceback

from agents.extractor_agent import (
    extract_text_from_pdf_tool,
    transcribe_audio_tool,
    upsert_to_qdrant_tool
)

from core.embeddings import semantic_chunk_text, get_embeddings
from core.crew_pipeline import RAGQueryTask
from core.crew_rag_pipeline_conditional import ConditionalRAGCrew
from config import QDRANT_COLLECTION
from core.crew_rag_pipeline_conditional import (
    ConditionalRAGCrew,
    extractor_agent,
    answer_agent,
    improver_agent,
    search_fallback_agent,
    rag_agent,
    evaluator_agent,
    retrieve_task,
    draft_task,
    improve_task,
    webfallback_task,
    evaluate_task
)


# ========================================
# Helper to print clean separators
# ========================================
def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(" " + title)
        print("=" * 60)
    else:
        print("=" * 60)


# ========================================
# MAIN DEBUG APP
# ========================================
def main():
    separator("DEBUG RAG PIPELINE (Console Mode)")

    # -------------------------
    # 1. Ask for file path
    # -------------------------
    file_path = input("Enter full path of PDF or audio file: ").strip()

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    file_ext = os.path.splitext(file_path)[1].lower()

    separator("EXTRACTING CONTENT")

    # -------------------------
    # 2. Extract Content
    # -------------------------
    try:
        if file_ext == ".pdf":
            with open(file_path, "rb") as f:
                text_content = extract_text_from_pdf_tool.run(f)

        elif file_ext in [".mp3", ".wav", ".m4a"]:
            with open(file_path, "rb") as f:
                text_content = transcribe_audio_tool.run(f)

        else:
            print(f"Unsupported file type: {file_ext}")
            sys.exit(1)

        print("\nExtracted Text (first 500 chars):")
        print(text_content[:500])

    except Exception as e:
        print("Extraction failed:")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------
    # 3. Chunk & Embed
    # -------------------------
    separator("CHUNKING & EMBEDDING")

    chunks = semantic_chunk_text(text_content)
    print(f"\nTotal chunks: {len(chunks)}")

    embeddings = get_embeddings(chunks)
    print("Embeddings generated.")

    metadata = [{"source": os.path.basename(file_path)} for _ in chunks]

    # -------------------------
    # 4. Qdrant Upsert
    # -------------------------
    separator("UPSERT TO LOCAL QDRANT")

    try:
        upsert_to_qdrant_tool.run(
            QDRANT_COLLECTION,
            chunks,
            metadata,
            embeddings
        )
        print("Successfully stored in Qdrant.")

    except Exception as e:
        print("Failed to upsert into Qdrant:")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------
    # 5. Query Loop
    # -------------------------
    separator("RAG CHAT MODE (type 'exit' to quit)")

    # task = RAGQueryTask(
    # description="Answer the user's question using retrieved context.",
    # expected_output="A helpful, accurate answer using RAG retrieval."
    # )


    crew = ConditionalRAGCrew(
    collection=QDRANT_COLLECTION,
    agents=[
        extractor_agent,
        answer_agent,
        improver_agent,
        search_fallback_agent,
        rag_agent,
        evaluator_agent
    ],
    tasks=[
        retrieve_task,
        draft_task,
        improve_task,
        webfallback_task,
        evaluate_task
    ]
)


    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            #result = task.run(query, QDRANT_COLLECTION)
            result = crew.kickoff(query)
            print("\nANSWER:")
            print(result["answer"])

            print("\nSources:", ", ".join(result["sources"]))
            print("Confidence:", result["confidence"])

        except Exception as e:
            print("Error while answering:")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
