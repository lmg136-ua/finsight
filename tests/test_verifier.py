import pytest
from unittest.mock import patch, MagicMock
from app.agents.verifier import verify_claims
from app.models.schemas import (
    GraphState, 
    RetrievedChunk, 
    ChunkMetadata, 
    VerificationResult, 
    ClaimVerification
)

@patch('app.agents.verifier.ChatPromptTemplate')
@patch('app.agents.verifier.get_llm')
def test_verify_claims_supported(mock_get_llm, mock_cpt):
    expected_result = VerificationResult(
        verifications=[
            ClaimVerification(
                claim="Microsoft revenue in FY24 was $245.1B", 
                status="SUPPORTED", 
                explanation="Matches Document 1 page 42"
            )
        ],
        all_supported=True,
        feedback=""
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected_result
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    mock_cpt.from_messages.return_value = mock_prompt

    state = {
        "draft_answer": "Microsoft revenue in FY24 was $245.1B [msft.pdf, p. 42].",
        "retrieved_chunks": [
            RetrievedChunk(
                text="Total revenue was $245,120 million for fiscal year 2024.",
                metadata=ChunkMetadata(
                    company="Microsoft",
                    document_type="10-K",
                    year=2024,
                    page=42,
                    source_filename="msft.pdf"
                )
            )
        ]
    }

    res = verify_claims(state)
    assert res["verification_result"].all_supported is True
    assert len(res["verification_result"].verifications) == 1
    assert res["verification_result"].verifications[0].status == "SUPPORTED"
