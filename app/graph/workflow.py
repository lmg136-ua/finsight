from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from app.models.schemas import GraphState, RetrievedChunk
from app.agents.planner import plan_query
from app.agents.analyst import analyze_and_draft
from app.agents.verifier import verify_claims
from app.retrieval.retriever import global_retriever as retriever

def retrieve_node(state: GraphState) -> GraphState:
    plan = state.get("plan")
    question = state["question"]
    
    all_chunks: List[RetrievedChunk] = []
    traces = list(state.get("retrieval_traces") or [])
    attempt = state.get("iteration_count", 0) + 1
    
    if plan and plan.tasks:
        for task in plan.tasks:
            query = f"{task.description} {' '.join(task.keywords)}"
            
            # Check if document exists before retrieval
            filter_dict = {}
            if task.company: 
                filter_dict["company"] = task.company
            if task.year: 
                filter_dict["year"] = int(task.year)
            
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
                
            # Retrieve top 3 per entity
            chunks = retriever.search(query, company=task.company, year=task.year, top_k=3)
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
        fallback_chunks = retriever.search(question, top_k=3)
        all_chunks.extend(fallback_chunks)
        
    # Fair context selection across entities (prevent one company from dominating context)
    seen = set()
    chunks_by_company: Dict[str, List[RetrievedChunk]] = {}
    for chunk in all_chunks:
        comp = chunk.metadata.company or "General"
        if comp not in chunks_by_company:
            chunks_by_company[comp] = []
        if chunk.text not in seen:
            seen.add(chunk.text)
            chunks_by_company[comp].append(chunk)
            
    num_companies = len(chunks_by_company)
    max_per_comp = max(1, 4 // max(1, num_companies))
    
    fair_chunks: List[RetrievedChunk] = []
    for comp, comp_chunks in chunks_by_company.items():
        fair_chunks.extend(comp_chunks[:max_per_comp])
        
    # Fill remaining capacity up to 4 pages (approx 2,000 tokens for strict TPM safety)
    if len(fair_chunks) < 4:
        for comp, comp_chunks in chunks_by_company.items():
            for c in comp_chunks[max_per_comp:]:
                if len(fair_chunks) >= 4:
                    break
                fair_chunks.append(c)
                
    return {"retrieved_chunks": fair_chunks, "retrieval_traces": traces}

def finalize_answer(state: GraphState) -> GraphState:
    # Accept the verified/draft answer as the final response
    return {"final_answer": state.get("draft_answer", "No answer generated.")}

def should_loop(state: GraphState) -> str:
    ver_res = state.get("verification_result")
    iteration = state.get("iteration_count", 0)
    
    if not ver_res or ver_res.all_supported or iteration >= 2:
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

# Lazy compiled graph singleton
_app_graph = None

def get_app_graph():
    global _app_graph
    if _app_graph is None:
        _app_graph = build_graph()
    return _app_graph

class _LazyGraphProxy:
    """Proxy object that delegates all calls to get_app_graph()."""
    def __getattr__(self, name):
        return getattr(get_app_graph(), name)
    def invoke(self, *args, **kwargs):
        return get_app_graph().invoke(*args, **kwargs)

app_graph = _LazyGraphProxy()
