from app.models.schemas import GraphState, VerificationResult
from app.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

def verify_claims(state: GraphState) -> GraphState:
    """
    Verifies the draft answer against the retrieved context.
    """
    draft = state["draft_answer"]
    chunks = state["retrieved_chunks"]
    
    context = ""
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        context += f"Source: {meta.source_filename}, Page {meta.page}\n"
        context += f"Content: {chunk.text}\n\n"
        
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(VerificationResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Evidence Verifier in a financial research system.\n"
                   "Your job is to read a draft answer and the raw source context, and verify if the important factual and numerical claims in the draft are supported by the context.\n"
                   "For each important claim, determine if it is SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or INSUFFICIENT_EVIDENCE.\n"
                   "If the draft claims that some information is missing from the context (e.g. 'Microsoft data is missing'), flag this as INSUFFICIENT_EVIDENCE so the system can trigger a new search.\n"
                   "If any important claim is UNSUPPORTED, PARTIALLY_SUPPORTED, or INSUFFICIENT_EVIDENCE, set `all_supported` to False, and provide feedback on what needs to be fixed or retrieved.\n"
                   "Context:\n{context}"),
        ("user", "Draft Answer:\n{draft}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "draft": draft})
    
    return {"verification_result": result}
