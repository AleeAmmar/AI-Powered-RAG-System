# app.py
import os
import streamlit as st
from agents.extractor_agent import (
    extract_text_from_pdf_tool,
    transcribe_audio_tool,
    upsert_to_qdrant_tool,
    extractor_agent
)
from core.embeddings import semantic_chunk_text, get_embeddings
from core.crew_rag_pipeline_conditional import (
    ConditionalRAGCrew,
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
from config import QDRANT_COLLECTION

# ---------------------------
# Initialize Crew
# ---------------------------
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot with PDF/Audio Upload")

# Sidebar: file upload
st.sidebar.header("Upload Documents (PDF / MP3 / WAV)")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "mp3", "wav", "m4a"])

# ---------------------------
# Handle file upload
# ---------------------------
if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

    try:
        if file_ext == ".pdf":
            text_content = extract_text_from_pdf_tool.run(uploaded_file)
        elif file_ext in [".mp3", ".wav", ".m4a"]:
            text_content = transcribe_audio_tool.run(uploaded_file)
        else:
            st.error("Unsupported file type")
            text_content = None

        if text_content:
            st.sidebar.info(f"Extracted {len(text_content)} characters from the file")

            # Chunk & Embed
            chunks = semantic_chunk_text(text_content)
            embeddings = get_embeddings(chunks)
            metadata = [{"source": uploaded_file.name} for _ in chunks]

            # Upsert to Qdrant
            upsert_to_qdrant_tool.run(
                QDRANT_COLLECTION, chunks, metadata, embeddings
            )
            st.sidebar.success(f"Inserted {len(chunks)} chunks into Qdrant.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------
# Chat interface
# ---------------------------
st.header("Ask a Question")
query = st.text_input("Type your question here:")

if st.button("Submit") and query:
    # placeholders for real-time updates
    status_placeholder = st.empty()
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    confidence_placeholder = st.empty()
    origin_placeholder = st.empty()

    try:
        origin = "RAG"  # default

        # -------------------------
        # Stage 1: Retrieve
        # -------------------------
        status_placeholder.info("Searching RAG...")
        retrieve_out = crew.retrieve_task_fn(query)
        context = retrieve_out["context"]
        search_results = retrieve_out["search_results"]
        r_conf = max((d.get("score", 0.0) for d in search_results), default=0.0)

        # -------------------------
        # Stage 2: Draft
        # -------------------------
        status_placeholder.info("Drafting answer...")
        draft_answer = crew.draft_task_fn(query, context)

        # -------------------------
        # Stage 3: Conditional Improve
        # -------------------------
        if r_conf < 0.6:
            status_placeholder.info("Improving answer...")
            improved = crew.improve_task_fn(query, context, draft_answer)
            if improved:
                draft_answer = improved
                origin = "Improved Answer"
            else:
                status_placeholder.info("Performing Web Search fallback...")
                fallback_done = crew.webfallback_task_fn(query)
                if fallback_done:
                    origin = "Web Search Fallback"
                    # Re-run retrieval + draft after inserting new web knowledge
                    retrieve_out = crew.retrieve_task_fn(query)
                    context = retrieve_out["context"]
                    search_results = retrieve_out["search_results"]
                    draft_answer = crew.draft_task_fn(query, context)

        # -------------------------
        # Stage 4: Evaluate
        # -------------------------
        confidence = crew.evaluate_task_fn(query, draft_answer, search_results, context)

        # -------------------------
        # Display results
        # -------------------------
        status_placeholder.success("Done")
        answer_placeholder.subheader("Answer")
        answer_placeholder.write(draft_answer)

        sources_placeholder.subheader("Sources")
        sources_placeholder.write([d["payload"].get("source", "unknown") for d in search_results])

        confidence_placeholder.subheader("Confidence")
        confidence_placeholder.write(confidence)

        origin_placeholder.subheader("Answer Origin")
        origin_placeholder.write(origin)

    except Exception as e:
        status_placeholder.error(f"Error while answering: {e}")
