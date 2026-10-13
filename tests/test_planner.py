import pytest
from unittest.mock import patch, MagicMock
from app.agents.planner import plan_query
from app.models.schemas import ResearchPlan, ResearchTask

@patch('app.agents.planner.get_available_metadata', return_value="Available companies: Microsoft, Alphabet. Available years: 2024.")
@patch('app.agents.planner.ChatPromptTemplate')
@patch('app.agents.planner.get_llm')
def test_plan_query_multi_company(mock_get_llm, mock_cpt, mock_meta):
    expected_plan = ResearchPlan(
        tasks=[
            ResearchTask(
                description="Retrieve Alphabet FY24 revenue and margins",
                company="Alphabet",
                year=2024,
                keywords=["Revenues", "Consolidated Statements of Income"]
            ),
            ResearchTask(
                description="Retrieve Microsoft FY24 revenue and margins",
                company="Microsoft",
                year=2024,
                keywords=["Revenue", "Consolidated Statements of Income"]
            )
        ],
        reasoning="Multi-company comparison query requires separate retrieval passes."
    )
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected_plan
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    mock_cpt.from_messages.return_value = mock_prompt

    state = {
        "question": "Compare revenue for Microsoft and Alphabet in 2024.",
        "chat_history": []
    }

    res = plan_query(state)
    assert "plan" in res
    tasks = res["plan"].tasks
    assert len(tasks) == 2
    assert tasks[0].company == "Alphabet"
    assert tasks[1].company == "Microsoft"
