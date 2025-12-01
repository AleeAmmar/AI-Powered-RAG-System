# agents/extractor_agent.py
from crewai import Agent
from crewai.tools import tool
import os
import tempfile

import fitz  # PyMuPDF
import whisper

from core.qdrant_utils import upsert_to_qdrant as _upsert_to_qdrant
from core.embeddings import get_embeddings


# ------------------------------------------------------
# Load Whisper globally ONCE (fastest approach)
# ------------------------------------------------------
WHISPER_MODEL = whisper.load_model("base")


# ------------------------------------------------------
# PDF Extraction (for uploaded files)
# ------------------------------------------------------
@tool("extract_text_from_pdf")
def extract_text_from_pdf_tool(uploaded_file) -> str:
    """Extract text from a PDF uploaded via Streamlit."""
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    text = ""
    with fitz.open(temp_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += f"\n--- Page {page_num} ---\n"
            text += page.get_text("text")

    os.remove(temp_path)
    return text.strip()


# ------------------------------------------------------
# Audio Transcription (using Whisper)
# ------------------------------------------------------
@tool("transcribe_audio")
def transcribe_audio_tool(uploaded_file) -> str:
    """Transcribe uploaded audio using Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    result = WHISPER_MODEL.transcribe(temp_path)
    os.remove(temp_path)
    return result["text"].strip()


# ------------------------------------------------------
# Upsert to Qdrant
# ------------------------------------------------------
@tool("upsert_to_qdrant")
def upsert_to_qdrant_tool(collection: str, texts: list, metadatas: list, embeddings: list):
    """Upserts text chunks and embeddings into a Qdrant collection."""
    return _upsert_to_qdrant(collection, texts, metadatas, embeddings)


# ------------------------------------------------------
# Extractor Agent
# ------------------------------------------------------
extractor_agent = Agent(
    name="Extractor",
    role="extractor",
    goal="Extract text from PDFs or audio files and store embeddings into Qdrant.",
    backstory="Handles ingestion for the RAG system: extraction, transcription, chunking, and vector DB insertion.",
    tools=[extract_text_from_pdf_tool, transcribe_audio_tool, upsert_to_qdrant_tool],
)
