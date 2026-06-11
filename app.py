import streamlit as st
from rag_engine import extract_text, chunk_text, build_index, retrieve, ask_gemini, get_api_key
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")

with st.sidebar:
    st.header("🔑 API Configuration")
    st.text_input(
        "Enter your Gemini API Key (Optional):",
        type="password",
        key="user_gemini_key",
        placeholder="AIzaSy..."
    )
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

api_key = get_api_key()

st.title("📄 RAG — Ask Your PDF")
st.caption("Upload a PDF and ask questions about it.")

if not api_key:
    st.warning("⚠️ Please provide a Gemini API key in the sidebar or app secrets to start using the app.")
    st.stop()

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    if "last_file" not in st.session_state or st.session_state.last_file != pdf_file.name:
        with st.spinner("Processing PDF..."):
            text = extract_text(pdf_file)
            chunks = chunk_text(text)
            index, _ = build_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.chat_history = []  # plain list, safe to store
            st.session_state.messages = []
            st.session_state.last_file = pdf_file.name

        st.success(f"Ready! Indexed {len(st.session_state.chunks)} chunks.")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask something about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            context = retrieve(question, st.session_state.chunks, st.session_state.index)
            answer, updated_history = ask_gemini(
                question,
                context,
                st.session_state.chat_history  # pass in current history
            )
            st.session_state.chat_history = updated_history  # save updated history back

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
# works