from langgraph.graph import StateGraph, START, END
from app.models.schemas import GraphState
from app.agents.planner import plan_query
from app.agents.analyst import analyze_and_draft
from app.agents.verifier import verify_claims
from app.retrieval.retriever import global_retriever as retriever

def retrieve_node(state: GraphState) -> GraphState:
    plan = state.get("plan")
    question = state["question"]
    
    all_chunks = []
    traces = state.get("retrieval_traces", [])
    attempt = state.get("iteration_count", 0) + 1
    
    if plan and plan.tasks:
        for task in plan.tasks:
            query = f"{task.description} {' '.join(task.keywords)}"
            
            # Check if document exists before retrieval
            filter_dict = {}
            if task.company: filter_dict["company"] = task.company
            if task.year: filter_dict["year"] = int(task.year)
            
            where_clause = None
            if len(filter_dict) == 1:
                where_clause = filter_dict
            elif len(filter_dict) > 1:
                where_clause = {"$and": [{k: v} for k, v in filter_dict.items()]}
                
            docs = retriever.vectorstore.get(where=where_clause, include=["metadatas"])
            matched_files = list(set([m["source_filename"] for m in docs["metadatas"] if m])) if docs and docs.get("metadatas") else []
            
            if not matched_files:
                traces.append({
                    "company": task.company,
                    "year": task.year,
                    "attempt": attempt,
                    "query": query,
                    "method": "hybrid (failed: not indexed)",
                    "candidates_retrieved": 0,
                    "matched_files": []
                })
                continue
                
            # Retrieve top 5 per company
            chunks = retriever.search(query, company=task.company, year=task.year, top_k=5)
            all_chunks.extend(chunks)
            
            traces.append({
                "company": task.company,
                "year": task.year,
                "attempt": attempt,
                "query": query,
                "method": "hybrid (bm25 + semantic)",
                "candidates_retrieved": len(chunks),
                "matched_files": matched_files
            })
    else:
        fallback_chunks = retriever.search(question, top_k=5)
        all_chunks.extend(fallback_chunks)
        
    # Deduplicate chunks based on text
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.text not in seen:
            seen.add(chunk.text)
            unique_chunks.append(chunk)
            
    # Keep up to 8 unique pages (approx 4000 tokens) to stay well under the 8000 TPM limit
    return {"retrieved_chunks": unique_chunks[:8], "retrieval_traces": traces}

def finalize_answer(state: GraphState) -> GraphState:
    # Accept the draft as the final answer
    return {"final_answer": state["draft_answer"]}

def should_loop(state: GraphState):
    ver_res = state["verification_result"]
    iteration = state.get("iteration_count", 0)
    
    if ver_res.all_supported or iteration >= 2:
        return "finalize"
    
    # If evidence is missing, replan and retrieve again
    if any(v.status == "INSUFFICIENT_EVIDENCE" for v in ver_res.verifications):
        return "re_plan"
        
    return "re_analyze"

def increment_iteration(state: GraphState) -> GraphState:
    return {"iteration_count": state.get("iteration_count", 0) + 1}

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("planner", plan_query)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("increment_iter", increment_iteration)
    workflow.add_node("analyst", analyze_and_draft)
    workflow.add_node("verifier", verify_claims)
    workflow.add_node("finalizer", finalize_answer)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "increment_iter")
    workflow.add_edge("increment_iter", "analyst")
    workflow.add_edge("analyst", "verifier")
    
    workflow.add_conditional_edges(
        "verifier",
        should_loop,
        {
            "finalize": "finalizer",
            "re_analyze": "increment_iter",
            "re_plan": "planner"
        }
    )
    
    workflow.add_edge("finalizer", END)
    
    return workflow.compile()

app_graph = build_graph()
