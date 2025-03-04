import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone using the new client interface
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define your index name and connect to it via the Pinecone client instance
index_name = "deepseek-papers-index"
index = pc.Index(index_name)

# Load the same embedding model used during ingestion
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(query, top_k=3):
    """
    Given a user query, generate its embedding and retrieve the most similar chunks from Pinecone.
    """
    # Generate query embedding
    query_embedding = model.encode([query]).tolist()[0]
    # Query Pinecone using keyword arguments: pass the vector using the "vector" parameter.
    query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    # Extract and return the text chunks from the metadata
    retrieved_chunks = [match['metadata']['text'] for match in query_response['matches']]
    return retrieved_chunks

if __name__ == "__main__":
    sample_query = "What are the key findings of the deepseek-v3 paper?"
    chunks = retrieve_relevant_chunks(sample_query)
    print("Retrieved chunks:")
    for chunk in chunks:
        print(chunk)
