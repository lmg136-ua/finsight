import json
import csv
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.workflow import app_graph
from app.models.schemas import GraphState

def run_evaluation():
    with open("evaluation/questions.json", "r") as f:
        questions = json.load(f)
        
    results = []
    
    for q in questions:
        print(f"Evaluating: {q['question']}")
        
        try:
            initial_state = {"question": q['question'], "chat_history": []}
            final_state = app_graph.invoke(initial_state)
            
            answer = final_state.get("final_answer", "")
            ver_result = final_state.get("verification_result")
            
            # Simple heuristic evaluation for demonstration
            hit_rate = 1 if final_state.get("retrieved_chunks") else 0
            
            if ver_result and ver_result.verifications:
                supported = sum(1 for v in ver_result.verifications if v.status == "SUPPORTED")
                faithfulness = supported / len(ver_result.verifications)
            else:
                faithfulness = 1.0 # default if no claims made
                
            results.append({
                "question": q['question'],
                "hit_rate": hit_rate,
                "faithfulness": faithfulness,
                "answer_length": len(answer),
                "iterations": final_state.get("iteration_count", 0)
            })
            
        except Exception as e:
            print(f"Error evaluating {q['question']}: {e}")
            results.append({
                "question": q['question'],
                "hit_rate": 0,
                "faithfulness": 0,
                "answer_length": 0,
                "iterations": 0
            })
            
    # Print summary
    print("\n--- Evaluation Summary ---")
    avg_hit_rate = sum(r["hit_rate"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    
    print(f"Average Hit Rate: {avg_hit_rate:.2f}")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}")
    
    with open("evaluation/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "hit_rate", "faithfulness", "answer_length", "iterations"])
        writer.writeheader()
        writer.writerows(results)
        
if __name__ == "__main__":
    run_evaluation()
