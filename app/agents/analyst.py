from app.models.schemas import GraphState
from app.agents.llm_factory import get_llm
from app.tools.financial_calculator import calculate_percentage_change, calculate_margin, calculate_ratio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Wrap our functions in LangChain tools
@tool
def percentage_change_tool(previous: float, current: float) -> str:
    """Calculate the percentage change between two numbers."""
    return calculate_percentage_change(previous, current)

@tool
def margin_tool(revenue: float, profit: float) -> str:
    """Calculate a profit or operating margin."""
    return calculate_margin(revenue, profit)

@tool
def ratio_tool(numerator: float, denominator: float) -> str:
    """Calculate a simple ratio."""
    return calculate_ratio(numerator, denominator)

tools = [percentage_change_tool, margin_tool, ratio_tool]

def analyze_and_draft(state: GraphState) -> GraphState:
    """
    Takes the retrieved chunks and answers the user's question.
    """
    question = state["question"]
    plan = state["plan"]
    chunks = state["retrieved_chunks"]
    feedback = ""
    if state.get("verification_result") and not state["verification_result"].all_supported:
        feedback = f"\n\nPREVIOUS ATTEMPT FEEDBACK (Please Fix):\n{state['verification_result'].feedback}"
    
    # Format context from chunks
    context = ""
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        context += f"--- Document {idx+1} ---\n"
        context += f"Source: {meta.source_filename}, Page {meta.page}, Company: {meta.company}, Year: {meta.year}\n"
        context += f"Content: {chunk.text}\n\n"
        
    llm = get_llm(temperature=0.2)
    
    agent = create_react_agent(llm, tools)
    
    prompt = f"""You are an expert financial analyst. Use the provided context to answer the user's question.
You must ALWAYS base your answers on the provided context. NEVER invent financial numbers.
Use the calculation tools when performing math. DO NOT do mental math for margins or growth rates.
Cite your sources using the format [Filename, p. X].
If the context is insufficient, state what is missing.

Below is the context:
{context}

Below is the research plan used to gather this context:
{plan.model_dump_json() if plan else 'No plan available.'}
{feedback}

User Question: {question}"""
    
    result = agent.invoke({"messages": [("user", prompt)]})
    
    return {"draft_answer": result["messages"][-1].content}
