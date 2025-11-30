"""
Quick test script - minimal version for testing
"""

import yaml
from src.vector_store import VectorStore

print("\n" + "="*80)
print("QUICK RAG TEST")
print("="*80)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vector store only
print("\nLoading vector index...")
vector_store = VectorStore(
    embedding_model_name=config['embedding']['model_name'],
    index_path=config['vector_db']['index_path'],
    use_faiss=True
)
vector_store.load_index()
print(f"✓ Loaded {len(vector_store.documents)} documents")

# Test retrieval only (no LLM)
question = "What are the top-rated electronics products?"

print(f"\nQuestion: {question}")
print("\nRetrieving documents...\n")

results = vector_store.search(question, top_k=3)

print("="*80)
print("TOP 3 RESULTS")
print("="*80)

for i, result in enumerate(results):
    print(f"\n[{i+1}] Score: {result['score']:.3f}")
    print("-"*80)
    text = result['document']['text']
    # Print first 200 chars
    print(text[:200] + "..." if len(text) > 200 else text)
    print()

print("="*80)
print("RETRIEVAL TEST COMPLETE!")
print("="*80)
print("\nThis shows the RAG system can retrieve relevant products.")
print("For LLM generation, use smaller models or reduce max_new_tokens.")
