import fitz # PyMuPDF extracts text from pdf
import faiss #stores vectors and finds similar vectors to the query
import numpy as np
from sentence_transformers import SentenceTransformer # converts text chunks into vector(locally)
from google import genai
import os
from dotenv import load_dotenv
import streamlit as st
import time


load_dotenv()

# 1. Initialize your local embedder (remains unchanged)
@st.cache_resource # Keeps the model cached in memory so it doesn't reload on every interaction
def load_embedder():
    return SentenceTransformer("all-MiniLM-l6-v2")

embedder = load_embedder()

# 2. Refactored function to prioritize the User's input over your own keys
def get_api_key():
    # Priority 1: Check if user typed it into the app sidebar
    if "user_gemini_key" in st.session_state and st.session_state["user_gemini_key"]:
        return st.session_state["user_gemini_key"]

    # Priority 2: Fallback to your own keys (Optional - remove if you want to block unauthorized access)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

    # 3. Create the UI Widget in the Sidebar to capture the key
with st.sidebar:
    st.header("API Configuration")

    # We store the input directly inside st.session_state["user_gemini_key"]
    st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        key="user_gemini_key",
        placeholder="AIzaSy..."
    )
    st.markdown("[Get a Gemini API Key](https://google.com)")

# 4. Fetch the final resolved key
api_key = get_api_key()

# 5. Stop the app execution if no key is found from ANY source
if not api_key:
    st.warning("⚠️ Please provide a Gemini API key in the sidebar to start using the RAG app.")
    st.stop()

# 6. Initialize the Gemini client safely using the resolved key
client = genai.Client(api_key=api_key)

def extract_text(pdf_file) -> str:
    """Extract all text from an uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range (0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index(chucks: list[str]):
    """Embed chunks and build FAISS vector index."""
    embeddings = embedder.encode(chucks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(query: str, chunks: list[str], index, top_k: int = 4) -> str:
    """Find most relevant chunks for query."""
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return "\n\n".join(chunks[i] for i in indices[0] if i < len(chunks))

def ask_gemini(question: str, context: str, history: list) -> tuple[str, list]:
    prompt = f"""You are a helpful assistant analysing a document the user has uploaded.

Use the retrieved context below to answer specific questions about the document's content.
For broad or general questions about the document (e.g. "what type of document is this?", 
"what is this used for?", "summarise this"), use your general knowledge combined with the context.
Only say you don't know if the question is genuinely unanswerable from both the context and general knowledge.

Retrieved context:
{context}

User question: {question}
"""
    # Add current question to history
    history.append({"role": "user", "parts": [{"text": prompt}]})

    for attempt in range(3):
        try:
            # Fresh client every call — avoids the "client closed" error
            fresh_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = fresh_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history
            )
            answer = response.text
            # Add Gemini's reply to history so next question has context
            history.append({"role": "model", "parts": [{"text": answer}]})
            return answer, history

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                # Remove the failed question from history so it doesn't corrupt future turns
                history.pop()
                return f"⚠️ Gemini unavailable. Try again.\n\n(Error: {e})", history