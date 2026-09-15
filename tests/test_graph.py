from app.graph.workflow import should_loop
from app.models.schemas import VerificationResult

def test_should_loop_all_supported():
    state = {
        "verification_result": VerificationResult(verifications=[], all_supported=True, feedback=""),
        "iteration_count": 1
    }
    assert should_loop(state) == "finalize"

def test_should_loop_not_supported_iter_1():
    state = {
        "verification_result": VerificationResult(verifications=[], all_supported=False, feedback="Missing something"),
        "iteration_count": 1
    }
    assert should_loop(state) == "re_analyze"

def test_should_loop_not_supported_iter_max():
    state = {
        "verification_result": VerificationResult(verifications=[], all_supported=False, feedback="Missing something"),
        "iteration_count": 2
    }
    assert should_loop(state) == "finalize"
