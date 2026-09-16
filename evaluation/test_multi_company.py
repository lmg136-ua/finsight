import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.workflow import app_graph
from app.retrieval.retriever import FinSightRetriever

def test_multi_company():
    print("Running Multi-Company Table Retrieval Test...")
    question = "What was the revenue, operating income, and net income for Alphabet and Microsoft in 2024 and 2023? Create a comparison table."
    
    initial_state = {
        "question": question,
        "chat_history": []
    }
    
    print(f"Question: {question}")
    final_state = app_graph.invoke(initial_state)
    
    retrieved_chunks = final_state.get("retrieved_chunks", [])
    
    # Check if both companies were retrieved
    companies = set(chunk.metadata.company for chunk in retrieved_chunks)
    print(f"\nCompanies retrieved: {companies}")
    
    if "Alphabet" in companies and "Microsoft" in companies:
        print("✅ SUCCESS: Both Microsoft and Alphabet chunks were retrieved.")
    else:
        print("❌ FAILED: Missing one or both companies in retrieval.")
        
    print("\nFinal Answer:")
    print(final_state.get("final_answer", ""))
    
if __name__ == "__main__":
    test_multi_company()