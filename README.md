# 📄 PDF RAG — Ask Your PDF

A Retrieval-Augmented Generation (RAG) web app that lets you upload any PDF and ask questions about it in natural language. Supports follow-up questions with full conversation memory.

🔗 **Live Demo:** [pdf-ragfi.streamlit.app](https://pdf-ragfi.streamlit.app)

---

## How It Works

```
PDF Upload → Text Extraction → Chunking → Vector Embeddings
→ FAISS Index → User Query → Similarity Search
→ Retrieved Context + Gemini 2.5 Flash → Answer
```

1. **Upload** a PDF — the app extracts and chunks the text
2. **Ask** any question — the app finds the most relevant chunks using vector similarity search
3. **Get answers** grounded in your document — with conversation memory for follow-up questions

---

## Features

- 📂 Upload any PDF and query its contents in natural language
- 🧠 Semantic search using vector embeddings — finds meaning, not just keywords
- 💬 Multi-turn conversation memory — ask follow-up questions naturally
- 🌐 General document questions supported — "what is this document?" works too
- ⚡ Powered by Gemini 2.5 Flash — fast and free tier friendly
- 🚀 Deployed on Streamlit Community Cloud

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Text extraction | `PyMuPDF (fitz)` | Parse text from PDF pages |
| Text chunking | Python | Split text into overlapping 500-word chunks |
| Embeddings | `sentence-transformers` | Convert chunks to semantic vectors locally |
| Vector search | `FAISS` | Find most relevant chunks for a query |
| LLM | `Google Gemini 2.5 Flash` | Generate natural language answers |
| UI | `Streamlit` | Web interface and session management |

---

## Run Locally

### Prerequisites
- Python 3.10+
- A free [Google AI Studio](https://aistudio.google.com) API key

### Setup

```bash
git clone https://github.com/your-username/pdf-rag.git
cd pdf-rag
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
pdf-rag/
├── app.py              # Streamlit UI and session management
├── rag_engine.py       # RAG pipeline (extraction, chunking, embeddings, retrieval, generation)
├── requirements.txt
└── .env                # local only — never committed
```

---

## What is RAG?

**Retrieval-Augmented Generation** solves a core limitation of LLMs — they don't know what's in *your* documents.

Instead of asking the LLM directly, RAG:
1. Searches your document for the most relevant passages
2. Passes those passages as context in the prompt
3. Asks the LLM to answer using only that context

This keeps answers grounded in your actual document rather than the model's general training data.

---

## Author

**Darsh Patel** — [github.com/darshfi](https://github.com/darshfi) · [linkedin.com/in/darsh-patel-fi](https://linkedin.com/in/darsh-patel-fi)
