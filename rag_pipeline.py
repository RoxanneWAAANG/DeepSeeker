from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_response(prompt):
    """
    Generate a response from the LLM using the ChatCompletion endpoint.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Change the model if needed
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def load_embeddings_from_sqlite(db_path="rag_db.sqlite"):
    """
    Load all embeddings from the SQLite database.
    Each row contains an id, a text chunk, and the embedding stored as a JSON string.
    Returns a list of tuples: (id, chunk, embedding as numpy array).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, chunk, embedding FROM embeddings")
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        id_, chunk, emb_json = row
        emb_list = json.loads(emb_json)
        emb_array = np.array(emb_list, dtype=np.float32)
        results.append((id_, chunk, emb_array))
    return results

def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def retrieve_relevant_chunks(query, model, db_path="rag_db.sqlite", top_k=3):
    """
    Given a query, generate its embedding and retrieve the top_k relevant text chunks
    from the SQLite database by computing cosine similarity.
    """
    # Generate the query's embedding using the same model used for indexing
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    query_embedding /= np.linalg.norm(query_embedding)
    
    # Load stored embeddings from SQLite
    stored_embeddings = load_embeddings_from_sqlite(db_path)
    similarities = []
    for id_, chunk, emb in stored_embeddings:
        norm = np.linalg.norm(emb)
        emb_norm = emb / norm if norm != 0 else emb
        sim = np.dot(query_embedding, emb_norm)
        similarities.append((id_, chunk, sim))
    
    # Sort by similarity score (highest first) and return the top_k chunks
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

def build_prompt(query, retrieved_chunks):
    """
    Build a prompt by combining the retrieved context chunks and the user query.
    """
    # Combine retrieved chunks into a single context string.
    context_text = "\n\n".join([chunk for _, chunk, _ in retrieved_chunks])
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    return prompt

def rag_pipeline(query, model, db_path="rag_db.sqlite", top_k=3):
    """
    Complete RAG pipeline: retrieve context from SQLite, build the prompt,
    and generate a response using the LLM.
    """
    retrieved = retrieve_relevant_chunks(query, model, db_path, top_k)
    prompt = build_prompt(query, retrieved)
    answer = generate_response(prompt)
    return answer

if __name__ == "__main__":
    # Initialize the SentenceTransformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example query input.
    query = "What are the key findings of deepseek v2?"
    
    # Run the RAG pipeline.
    answer = rag_pipeline(query, model)
    print("LLM Response:", answer)

