from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Retrieval Schemas
class ChunkMetadata(BaseModel):
    company: str
    document_type: str
    year: int
    page: int
    section: Optional[str] = None
    source_filename: str

class RetrievedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata
    score: Optional[float] = None

# Planner Schemas
class ResearchTask(BaseModel):
    description: str = Field(description="Description of the information to retrieve or analyze")
    company: Optional[str] = Field(None, description="Company this task relates to, if specific")
    keywords: List[str] = Field(default_factory=list, description="Keywords for retrieval")

class ResearchPlan(BaseModel):
    tasks: List[ResearchTask] = Field(description="List of tasks to execute")
    reasoning: str = Field(description="Reasoning for this plan")

# Analysis and Verifier Schemas
class Citation(BaseModel):
    source_filename: str = Field(description="Name of the source document")
    page: int = Field(description="Page number")
    company: str = Field(description="Company related to the citation")
    quote: str = Field(description="Exact quote or very close paraphrase from the text")

class FactualClaim(BaseModel):
    claim: str = Field(description="The factual claim being made")
    citations: List[Citation] = Field(description="Citations supporting this claim")

class ClaimVerification(BaseModel):
    claim: str = Field(description="The claim being verified")
    status: str = Field(description="SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED")
    explanation: str = Field(description="Explanation for the verification status")

class VerificationResult(BaseModel):
    verifications: List[ClaimVerification] = Field(description="List of verified claims")
    all_supported: bool = Field(description="True if all important claims are SUPPORTED")
    feedback: str = Field(description="Feedback for the analyst if not all claims are supported")

# Graph State
class GraphState(TypedDict):
    question: str
    plan: Optional[ResearchPlan]
    retrieved_chunks: List[RetrievedChunk]
    draft_answer: Optional[str]
    verification_result: Optional[VerificationResult]
    final_answer: Optional[str]
    iteration_count: int
    chat_history: List[Dict[str, str]]
