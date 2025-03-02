import os
from embeddings import load_preprocessed_chunks, generate_embeddings

# Import the new Pinecone client and ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone using the new client interface
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index details
index_name = "deepseek-papers-index"
embedding_dim = 384

# Check if the index exists; if not, create it using a ServerlessSpec.
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
# Connect to the created index.
index = pc.Index(index_name)

# --- Load Preprocessed Chunks ---
chunks = load_preprocessed_chunks(folder="preprocessed_data")
print(f"Loaded {len(chunks)} chunks from preprocessed_data folder.")

# --- Generate Embeddings for the Chunks ---
embeddings, model = generate_embeddings(chunks)
print("Generated embeddings shape:", embeddings.shape)

# --- Prepare Data and Upsert into Pinecone ---
upsert_data = []
for i, emb in enumerate(embeddings):
    item_id = str(i)
    vector = emb.tolist()
    metadata = {
        "text": chunks[i],
        "chunk_id": i
    }
    upsert_data.append((item_id, vector, metadata))

# Upsert in batches using the "vectors" parameter.
batch_size = 100
for i in range(0, len(upsert_data), batch_size):
    batch = upsert_data[i:i+batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted batch {i // batch_size + 1}")

print(f"Successfully upserted {len(upsert_data)} vectors into Pinecone.")