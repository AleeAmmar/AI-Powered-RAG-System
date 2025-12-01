# AI-Powered Hybrid RAG System  
A complete end-to-end **Retrieval-Augmented Generation (RAG)** pipeline with:

- PDF Ingestion  
- Audio Transcription (Whisper)  
- Vector Search (Qdrant)  
- Web Search Fallback  
- Multi-Agent Workflow (CrewAI)  
- Streamlit Chat Interface  
- Conditional Human-like Reasoning Pipeline (Retrieve → Draft → Improve → Search → Final Evaluate)

---

## Features

### **Document Ingestion**
- Upload **PDFs** → Extract text using PyMuPDF  
- Upload **audio (mp3/wav/m4a)** → Transcribe using Whisper  
- Semantic chunking and embedding  
- Automatic vector storage to **Qdrant**

---

## Multi-Agent System (CrewAI)
This system uses multiple cooperating AI agents:

| Agent | Responsibility |
|-------|----------------|
| **Extractor Agent** | Extracts PDF/audio text & saves chunks in Qdrant |
| **Retriever Agent (internal)** | Retrieves best matching chunks |
| **RAG Answer Agent** | Generates answer from retrieved context |
| **Improver Agent** | Improves the draft answer when confidence is low |
| **Web Search Fallback Agent** | Performs web search when RAG confidence < threshold |
| **Evaluator Agent** | Computes confidence score on final answer |

---

## **Hybrid RAG Logic**
The system dynamically chooses the answer strategy:

1. **High confidence RAG → Answer**  
2. **Medium confidence → Improve via LLM**  
3. **Low confidence → Web Search → Save result to Qdrant → Retry RAG**

This achieves:
- Higher accuracy  
- Lower hallucinations  
- Continuous knowledge growth  

---

## 🖥 Streamlit Interface
The frontend includes:
- Chat-style query box  
- PDF & audio upload sidebar  
- Real-time pipeline status  
  - “Searching RAG…”  
  - “Improving answer…”  
  - “Running Web Search…”  
- Final answer  
- Retrieved Sources  
- Confidence score  
- Answer origin (RAG / Improved / WebSearch)

---

## 🚀 How to Use the RAG + CrewAI System

Follow these steps to set up and run the system:

### 1️⃣ Setup Project Environment

1. Open the project folder in **Visual Studio Code** (or your preferred IDE).
2. Create a new **Python virtual environment** with Python 3.11:

   ```bash
   python -m venv venv
   ```
3. Activate the environment:

   * **Windows:** `venv\Scripts\activate`
   * **Linux / Mac:** `source venv/bin/activate`
4. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

### 2️⃣ Configure the System

1. Open `config.py`.
2. Set the following values:

   * **LMStudio URL:** URL of your local LM Studio instance
   * **Qdrant URL:** Local Qdrant server URL
   * **LMStudio Model:** Model name you want to use (e.g., `ggml-model.bin`)
   * **SERPAPI Key:** Your SERPAPI API key (for web search fallback)

### 3️⃣ Run Supporting Services

* **LM Studio:** Start your local LM Studio instance.
* **Qdrant:** Run Qdrant locally (e.g., via Docker):

  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

### 4️⃣ Launch the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### 5️⃣ Using the App

* **Upload PDF or Audio files** to add new documents to the RAG knowledge base.

* **Type your query** in the search box to retrieve answers.

* The app will indicate **where the answer came from**:

  * **RAG** retrieval
  * **Improved answer**
  * **Web search fallback**

* Each answer also shows **confidence score and sources**.

### ✅ Notes

* Make sure **LM Studio** and **Qdrant** are running before using the app.
* Uploaded documents are automatically **chunked, embedded, and stored in Qdrant**.
* The system supports dynamic knowledge enrichment through **web search fallback**.


## 🗂 Project Structure
```project/
│
├── app.py # Streamlit application (chat UI + file upload)
│
├── agents/ # CrewAI Agents and tools
│ ├── extractor_agent.py # Extract PDF/audio, chunk, embed, upsert to Qdrant
│ ├── search_fallback_agent.py# Web search fallback agent
│ ├── rag_agent.py # RAG retrieval agent
│ ├── improver_agent.py # Answer improvement agent
│ ├── answer_agent.py # Draft answer LLM agent
│ ├── evaluator_agent.py # Confidence evaluation agent
│
├── core/ # Core pipeline logic and utilities
│ ├── crew_pipeline.py # Base CrewAI task abstractions
│ ├── crew_rag_pipeline_conditional.py # Conditional RAG Crew implementation
│ ├── embeddings.py # Chunking & embedding functions
│ ├── qdrant_utils.py # Qdrant upsert & query helper functions
│
├── config.py # Configuration (e.g., Qdrant collection name)
│
└── README.md # Project overview, architecture, instructions```
