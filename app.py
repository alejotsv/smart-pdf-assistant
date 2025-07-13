import streamlit as st
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from rag_engine.llm_interface import get_answer_from_question
from dotenv import load_dotenv

try:
    from dotenv import load_dotenv
    load_dotenv()
# Not needed in Streamlit Cloud
except ImportError:
    pass

st.set_page_config(page_title="Smart PDF RAG Assistant", layout="centered")
st.title("📄 Smart PDF RAG Assistant")

st.markdown("""
Ask a question about one of three loaded documents:

- 📜 **U.S. Constitution**
- 🛡️ **GDPR Regulation**
- 🦾 **Attention Is All You Need** (Transformer paper)

This app uses **agentic AI routing** to classify your question and retrieve relevant text chunks using **local embeddings**, then passes that context to GPT-3.5 to generate an answer.

Try asking:
- *“What is the Fourth Amendment about?”*
- *“What rights do data subjects have?”*
- *“What is positional encoding in transformers?”*
""")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

question = st.text_input("Ask a question about the Constitution, GDPR, or Transformers:")

if question:
    with st.spinner("Thinking..."):
        try:
            result = get_answer_from_question(question, client, model)
            st.markdown(f"**Topic:** {result['visible_topic']}")
            st.markdown(f"**Answer:** {result['answer']}")

            with st.expander("🔍 Retrieved chunks (context)", expanded=False):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}:**\n{chunk}\n")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
