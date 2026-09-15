from app.models.schemas import GraphState, ResearchPlan
from app.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

def plan_query(state: GraphState) -> GraphState:
    """
    Analyzes the question and breaks it down into research tasks.
    """
    question = state["question"]
    llm = get_llm(temperature=0.0)
    
    # We use structured output for the LLM
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial researcher. Your job is to take a complex user question and break it down into a clear research plan.\n"
                   "Identify the companies involved, the specific metrics or information required, and decompose complex questions into atomic search tasks.\n"
                   "Ensure you generate appropriate keywords for semantic search retrieval."),
        ("user", "{question}")
    ])
    
    chain = prompt | structured_llm
    plan = chain.invoke({"question": question})
    
    return {"plan": plan}
