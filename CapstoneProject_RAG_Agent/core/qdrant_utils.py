from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Configuration
# ---------------------------
QDRANT_COLLECTION = "my_collection"
QDRANT_URL = "http://localhost:6333"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
collections = [c.name for c in client.get_collections().collections]

if QDRANT_COLLECTION not in collections:
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# ---------------------------
# Custom Document class
# ---------------------------
class CustomDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata
        self.id = str(uuid4())

# ---------------------------
# Upsert function
# ---------------------------
def upsert_to_qdrant(collection, texts, metadatas, embeddings):
    """
    Upserts documents into Qdrant.
    """
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embeddings[i],
            payload={**metadatas[i], "text": texts[i]}
        )
        for i in range(len(texts))
    ]
    client.upsert(
        collection_name=collection,
        points=points
    )

# ---------------------------
# Query function
# ---------------------------
def qdrant_query(query, topk=5, collection="my_collection"):
    query_vector = embedding_model.embed_query(query)
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=topk
    )
    
    hits = []
    for hit in response.points:  # <-- use .points
        hits.append({
            "payload": hit.payload,
            "score": hit.score or 0.0
        })
    return hits