import os
import glob
import json
import sqlite3
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2', batch_size=32, device=None):
    """
    Generate normalized embeddings for a list of text chunks.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    # Generate embeddings with batch processing.
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    
    # Normalize embeddings.
    embeddings_tensor = torch.tensor(embeddings)
    normalized_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
    normalized_embeddings = normalized_tensor.numpy()
    return normalized_embeddings, model

def load_preprocessed_chunks(folder="preprocessed_data"):
    """
    Load preprocessed text chunks from all text files in the specified folder.
    Each file is assumed to have multiple chunks separated by two newlines.
    """
    all_chunks = []
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if data:
                # Split chunks by double newlines.
                chunks = [chunk.strip() for chunk in data.split("\n\n") if chunk.strip()]
                all_chunks.extend(chunks)
    return all_chunks

def save_embeddings_to_sqlite(db_path="rag_db.sqlite", chunks=None, embeddings=None):
    """
    Save each text chunk and its embedding (as a JSON string) into an SQLite database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create table if it doesn't exist.
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT,
            embedding TEXT
        )
    """)
    
    # Insert each chunk with its embedding.
    for chunk, emb in zip(chunks, embeddings):
        emb_json = json.dumps(emb.tolist())  # Convert numpy array to list, then to JSON string.
        c.execute("INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)", (chunk, emb_json))
    
    conn.commit()
    conn.close()
    print(f"Saved {len(chunks)} embeddings into {db_path}.")

if __name__ == "__main__":
    # Load preprocessed chunks from the designated folder.
    chunks = load_preprocessed_chunks()
    print(f"Loaded {len(chunks)} chunks from preprocessed_data folder.")
    
    # Generate embeddings (using a smaller batch size for better GPU utilization, if available).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, model = generate_embeddings(chunks, batch_size=16, device=device)
    print("Embeddings shape:", embeddings.shape)
    
    # Save the generated embeddings and associated chunks into SQLite.
    save_embeddings_to_sqlite(chunks=chunks, embeddings=embeddings)
