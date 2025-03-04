# evaluation.py
import time
from sentence_transformers import SentenceTransformer
from rag_pipeline import rag_pipeline

def evaluate_system(queries, model, db_path="rag_db.sqlite", top_k=3):
    """
    Evaluate the RAG system over a list of sample queries.
    
    For each query, measure the total response time from retrieval to LLM generation,
    and store the query, answer, and elapsed time.
    
    Returns:
        results (list): A list of tuples (query, answer, elapsed_time).
    """
    results = []
    for query in queries:
        start_time = time.time()
        answer = rag_pipeline(query, model, db_path=db_path, top_k=top_k)
        elapsed = time.time() - start_time
        results.append((query, answer, elapsed))
    return results

if __name__ == "__main__":
    # Define a list of sample queries for evaluation.
    sample_queries = [
         "What are the key findings of the paper?",
         "Explain the impact of X on Y.",
         "Summarize the methodology used in this research.",
         "What future work is suggested by the authors?",
         "How does this study compare with previous research?"
    ]
    
    # Initialize the SentenceTransformer model (should match the one used in the RAG pipeline)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Run the evaluation over the sample queries.
    eval_results = evaluate_system(sample_queries, model)
    
    # Print out the results
    total_time = 0
    print("Evaluation Results:")
    print("="*60)
    for query, answer, elapsed in eval_results:
        print("Query: ", query)
        print("Answer:", answer)
        print("Response Time: {:.2f} seconds".format(elapsed))
        print("-"*60)
        total_time += elapsed
        
    avg_time = total_time / len(sample_queries)
    print("Average Response Time: {:.2f} seconds".format(avg_time))
