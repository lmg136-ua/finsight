import json
import csv
import sys
import os
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.workflow import app_graph
from app.models.schemas import GraphState

def run_evaluation():
    questions_file = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    results = []
    
    print("=" * 60)
    print("  FinSight Automated Benchmark Evaluation")
    print("=" * 60)
    
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Evaluating: {q['question']}")
        
        try:
            initial_state = {"question": q['question'], "chat_history": []}
            final_state = app_graph.invoke(initial_state)
            
            answer = final_state.get("final_answer", "")
            ver_result = final_state.get("verification_result")
            retrieved = final_state.get("retrieved_chunks", [])
            
            # Hit Rate: Was context retrieved successfully?
            hit_rate = 1.0 if len(retrieved) > 0 else 0.0
            
            # Faithfulness: Proportion of claims supported by citations
            if ver_result and ver_result.verifications:
                supported = sum(1 for v in ver_result.verifications if v.status == "SUPPORTED")
                faithfulness = round(supported / len(ver_result.verifications), 2)
            else:
                faithfulness = 1.0
                
            has_citations = 1 if ("[" in answer and "p." in answer) or ("Document" in answer) else 0
            
            print(f"  -> Hit Rate: {hit_rate * 100:.0f}% | Faithfulness: {faithfulness * 100:.0f}% | Chunks: {len(retrieved)} | Iterations: {final_state.get('iteration_count', 0)}")
            
            results.append({
                "question": q['question'],
                "hit_rate": hit_rate,
                "faithfulness": faithfulness,
                "answer_length": len(answer),
                "has_citations": has_citations,
                "iterations": final_state.get("iteration_count", 0)
            })
            
            time.sleep(3)
        except Exception as e:
            print(f"  -> Error evaluating question: {e}")
            results.append({
                "question": q['question'],
                "hit_rate": 0.0,
                "faithfulness": 0.0,
                "answer_length": 0,
                "has_citations": 0,
                "iterations": 0
            })
            time.sleep(3)
            
    # Print summary
    print("\n" + "=" * 60)
    print("  Benchmark Summary Results")
    print("=" * 60)
    avg_hit_rate = sum(r["hit_rate"] for r in results) / len(results) if results else 0
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    
    print(f"Average Retrieval Hit Rate: {avg_hit_rate * 100:.1f}%")
    print(f"Average Answer Faithfulness: {avg_faithfulness * 100:.1f}%")
    print(f"Benchmark Questions Evaluated: {len(results)}")
    
    results_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "hit_rate", "faithfulness", "answer_length", "has_citations", "iterations"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {results_path}")
        
if __name__ == "__main__":
    run_evaluation()
