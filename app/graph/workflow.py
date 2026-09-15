from langgraph.graph import StateGraph, START, END
from app.models.schemas import GraphState
from app.agents.planner import plan_query
from app.agents.analyst import analyze_and_draft
from app.agents.verifier import verify_claims
from app.retrieval.retriever import FinSightRetriever

retriever = FinSightRetriever()

def retrieve_node(state: GraphState) -> GraphState:
    plan = state["plan"]
    question = state["question"]
    
    all_chunks = []
    
    if plan and plan.tasks:
        for task in plan.tasks:
            query = f"{task.description} {' '.join(task.keywords)}"
            chunks = retriever.search(query, company=task.company, top_k=3)
            all_chunks.extend(chunks)
    else:
        # Fallback if no plan
        all_chunks = retriever.search(question, top_k=5)
        
    # Deduplicate chunks based on text
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.text not in seen:
            seen.add(chunk.text)
            unique_chunks.append(chunk)
            
    # Keep top 10 unique chunks to avoid overflowing context
    return {"retrieved_chunks": unique_chunks[:10]}

def finalize_answer(state: GraphState) -> GraphState:
    # Accept the draft as the final answer
    return {"final_answer": state["draft_answer"]}

def should_loop(state: GraphState):
    ver_res = state["verification_result"]
    iteration = state.get("iteration_count", 0)
    
    if ver_res.all_supported or iteration >= 2:
        return "finalize"
    else:
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
            "re_analyze": "increment_iter"
        }
    )
    
    workflow.add_edge("finalizer", END)
    
    return workflow.compile()

app_graph = build_graph()
