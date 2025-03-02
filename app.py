import streamlit as st
from rag_pipeline import rag_pipeline
from sentence_transformers import SentenceTransformer

st.title("RAG Q&A System for DeepSeek Papers")

# Initialize the SentenceTransformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    with st.spinner("Processing your query..."):
        # Run the full RAG pipeline: retrieval, prompt building, and LLM response generation.
        answer = rag_pipeline(query, model)
    st.markdown("**Answer:**")
    st.write(answer)
