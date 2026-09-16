from app.models.schemas import GraphState, ResearchPlan
from app.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from app.retrieval.retriever import global_retriever as retriever

def get_available_metadata():
    try:
        docs = retriever.vectorstore.get(include=["metadatas"])
        if not docs or not docs.get("metadatas"):
            return "No documents indexed."
        companies = set()
        years = set()
        for meta in docs["metadatas"]:
            if meta:
                companies.add(meta.get("company", "Unknown"))
                years.add(str(meta.get("year", "Unknown")))
        return f"Available companies: {', '.join(companies)}. Available years: {', '.join(years)}."
    except:
        return "Metadata unavailable."

def plan_query(state: GraphState) -> GraphState:
    """
    Analyzes the question and breaks it down into research tasks.
    """
    question = state["question"]
    
    # Check if we are replanning due to missing evidence
    feedback = ""
    if state.get("verification_result") and not state["verification_result"].all_supported:
        prev_plan = state.get("plan")
        prev_keywords = []
        if prev_plan:
            for t in prev_plan.tasks:
                prev_keywords.extend(t.keywords)
                
        feedback = f"\n\nPREVIOUS ATTEMPT FAILED. FEEDBACK:\n{state['verification_result'].feedback}\n"
        feedback += f"You previously searched for: {', '.join(prev_keywords)}.\n"
        feedback += "You MUST generate DIFFERENT keywords than your previous attempt. If you searched for general terms before, search for exact table headings like 'Consolidated Statements of Income' to find the raw table."

    llm = get_llm(temperature=0.0)
    
    # We use structured output for the LLM
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    db_meta = get_available_metadata()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial researcher. Your job is to take a complex user question and break it down into a clear research plan.\n"
                   "Identify the companies involved, the specific metrics or information required, and decompose complex questions into atomic search tasks.\n"
                   f"DATABASE INFO: {db_meta}\n"
                   "CRITICAL INSTRUCTIONS:\n"
                   "1. If the user asks about multiple companies, you MUST create at least one separate ResearchTask for EACH company and set the `company` field explicitly matching an available company above EXACTLY.\n"
                   "2. You MUST set the `year` field explicitly to the target filing year matching an available year above EXACTLY.\n"
                   "3. For FY2024 vs FY2023 comparisons, DO NOT search for separate 2023 filings. The 2024 10-K already contains comparative 2023 figures. Search only once per company for the 2024 year.\n"
                   "4. When asking for revenue, operating income, net income, or margins, strongly prioritize exact section headings in your keywords like 'Consolidated Statements of Income', 'Statements of Operations', or 'Financial Statements'.\n"
                   "5. For Microsoft, use keywords like 'Revenue', 'Operating income', 'Net income'. For Alphabet, use 'Revenues', 'Income from operations', 'Net income'.\n"
                   "Ensure you generate precise keywords for hybrid search retrieval.{feedback}"),
        ("user", "{question}")
    ])
    
    chain = prompt | structured_llm
    plan = chain.invoke({"question": question, "feedback": feedback})
    
    return {"plan": plan}
