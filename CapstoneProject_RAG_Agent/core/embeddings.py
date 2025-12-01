# embeddings_utils.py
import nltk
nltk.download('punkt')  # corrected from 'punkt_tab'
from nltk import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------------------------------
# Local embedding model
# ----------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_embeddings(texts):
    """
    Generate embeddings locally using HuggingFaceEmbeddings.
    Supports single string or list of strings.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = embedding_model.embed_documents(texts)
    return embeddings

# ----------------------------------------------------
# Chunking text
# ----------------------------------------------------
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.
    This is semantically better than naive token splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# ----------------------------------------------------
# Optional: sentence-based chunking (semantic chunks)
# ----------------------------------------------------
def semantic_chunk_text(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current = [], []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [sentence], sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))
    return chunks
