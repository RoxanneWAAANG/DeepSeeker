import os
import glob
import torch
from sentence_transformers import SentenceTransformer


def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2', batch_size=32, device=None):
    """
    Generate embeddings for a list of text chunks using a SentenceTransformer model.
    
    Parameters:
      - chunks (list[str]): List of text chunks.
      - model_name (str): Pre-trained model name or path.
      - batch_size (int): Batch size for processing chunks.
      - device (str): Device to run the model on ('cuda' or 'cpu').
      
    Returns:
      - embeddings (numpy.ndarray): Array of normalized embeddings.
      - model (SentenceTransformer): The loaded embedding model.
    """
    # Use the specified device if provided, otherwise select automatically.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    # Generate embeddings with a progress bar and batch processing.
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    
    # Convert numpy array to torch tensor for normalization.
    embeddings_tensor = torch.tensor(embeddings)
    normalized_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    
    # Convert normalized tensor back to numpy array.
    normalized_embeddings = normalized_tensor.numpy()
    return normalized_embeddings, model

def load_preprocessed_chunks(folder="preprocessed_data"):
    """
    Load all preprocessed chunks from text files in the given folder.
    Each file should contain multiple chunks separated by two newlines.
    
    Parameters:
      - folder (str): Directory where preprocessed chunk text files are stored.
      
    Returns:
      - all_chunks (list[str]): List of all text chunks.
    """
    all_chunks = []
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if data:
                # Split by double newlines to separate chunks
                chunks = [chunk.strip() for chunk in data.split("\n\n") if chunk.strip()]
                all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    # Load preprocessed chunks from the folder.
    chunks = load_preprocessed_chunks()
    print(f"Loaded {len(chunks)} chunks from preprocessed_data folder.")
    
    # Generate embeddings with a smaller batch size for better GPU utilization
    # and specify device explicitly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, model = generate_embeddings(chunks, batch_size=16, device=device)
    print("Embeddings shape:", embeddings.shape)
