# search_rag.py
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_embeddings_from_sqlite(db_path="rag_db.sqlite"):
    """
    Load all rows from the SQLite database.
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

def search_query(query, model, db_path="rag_db.sqlite", top_k=3):
    """
    Given a query, generate its embedding, then search the SQLite database
    for the most similar text chunks based on cosine similarity.
    """
    # Generate query embedding.
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    query_embedding /= np.linalg.norm(query_embedding)
    
    # Load stored embeddings.
    stored_embeddings = load_embeddings_from_sqlite(db_path)
    
    # Compute similarity scores.
    similarities = []
    for id_, chunk, emb in stored_embeddings:
        emb_norm = emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb
        sim = np.dot(query_embedding, emb_norm)
        similarities.append((id_, chunk, sim))
    
    # Sort results by similarity (highest first) and return the top_k.
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    # Initialize the same SentenceTransformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Accept a query from the user.
    query = input("Enter your query: ")
    top_results = search_query(query, model, top_k=3)
    
    # Display the top retrieved text chunks.
    print("Top relevant chunks:")
    for id_, chunk, sim in top_results:
        print(f"ID: {id_}, Similarity: {sim:.4f}")
        print(f"Chunk: {chunk}\n")


