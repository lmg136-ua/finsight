from app.models.schemas import GraphState
from app.agents.llm_factory import get_llm
from app.tools.financial_calculator import (
    calculate_percentage_change, 
    calculate_margin, 
    calculate_ratio,
    calculate_cagr,
    format_financial_amount
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Wrap financial calculator functions in LangChain tools
@tool
def percentage_change_tool(previous: float, current: float) -> str:
    """Calculate the percentage change between two numbers (e.g. YoY growth)."""
    return calculate_percentage_change(previous, current)

@tool
def margin_tool(revenue: float, profit: float) -> str:
    """Calculate a profit or operating margin percentage."""
    return calculate_margin(revenue, profit)

@tool
def ratio_tool(numerator: float, denominator: float) -> str:
    """Calculate a financial ratio."""
    return calculate_ratio(numerator, denominator)

@tool
def cagr_tool(beginning: float, ending: float, periods: int) -> str:
    """Calculate Compound Annual Growth Rate (CAGR) across multiple periods."""
    return calculate_cagr(beginning, ending, periods)

@tool
def format_amount_tool(amount: float, currency: str = "$") -> str:
    """Format large currency values into readable units (e.g. $245.12B or $150.00M)."""
    return format_financial_amount(amount, currency)

tools = [percentage_change_tool, margin_tool, ratio_tool, cagr_tool, format_amount_tool]

def analyze_and_draft(state: GraphState) -> GraphState:
    """
    Takes the retrieved chunks and synthesizes an evidence-backed answer to the user's question.
    """
    question = state["question"]
    plan = state.get("plan")
    chunks = state.get("retrieved_chunks", [])
    feedback = ""
    if state.get("verification_result") and not state["verification_result"].all_supported:
        feedback = f"\n\nPREVIOUS ATTEMPT FEEDBACK (Please Address & Correct):\n{state['verification_result'].feedback}"
    
    # Format context from retrieved chunks with precise source attribution
    context = ""
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        context += f"--- Document {idx+1} ---\n"
        context += f"Source: {meta.source_filename}, Page {meta.page}, Company: {meta.company}, Year: {meta.year}\n"
        context += f"Content: {chunk.text}\n\n"
        
    llm = get_llm(temperature=0.1)
    
    agent = create_react_agent(llm, tools)
    
    prompt = f"""You are an elite Senior Financial Analyst conducting equity research on SEC filings (Form 10-K / 10-Q).
Your goal is to provide a rigorous, institutional-grade answer to the user's inquiry based strictly on the provided filing evidence.

OPERATIONAL PRINCIPLES:
1. STRICT FACTUAL ACCURACY: Base all financial figures strictly on the context. NEVER fabricate, estimate, or hallucinate metrics.
2. TOOL USAGE: When calculating margins, YoY changes, ratios, or CAGR, ALWAYS invoke the corresponding calculation tool. Do not perform mental math.
3. CITATION PROTOCOL: Every factual or quantitative claim MUST be attributed with exact citations in the format `[Filename, p. X]`.
4. FORMATTING:
   - For multi-company or multi-period queries, construct clear, beautifully aligned Markdown comparison tables.
   - Separate numbers with appropriate thousands commas and specify currency/units (e.g., in millions or billions).
   - Ensure clean spacing in markdown formatting (e.g., "**Revenue** in FY24" instead of "**Revenue**in FY24").
5. INSUFFICIENT EVIDENCE: If a required metric is genuinely absent from the context, state clearly: "The required evidence was not retrieved from the indexed filings." Do not invent numbers.

Context Chunks:
{context}

Research Plan:
{plan.model_dump_json() if plan else 'No plan available.'}
{feedback}

User Question: {question}"""
    
    result = agent.invoke({"messages": [("user", prompt)]})
    
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
        
    return {"draft_answer": content}
