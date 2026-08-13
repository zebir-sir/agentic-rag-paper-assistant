from agent.intent_planner import IntentPlan, RetrievalStep
from agent.retrieval_harness_policy import (
    compile_retrieval_contract,
    enforce_retrieval_contract,
    evaluate_retrieval_contract,
)


def _plan(*steps: RetrievalStep, required=None, preferred=None, blocked=None) -> IntentPlan:
    return IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=list(steps),
        max_tools=2,
        source_requirements={
            "required_source_types": list(required or []),
            "preferred_source_types": list(preferred or []),
        },
        answer_policy={"blocked_source_types": list(blocked or [])},
    )


def test_contract_keeps_local_paper_task_inside_local_sources():
    plan = _plan(
        RetrievalStep(tool="hybrid_search", query="selected paper method"),
        required=["local_kb"],
        preferred=["local_kb"],
    )
    contract = compile_retrieval_contract(
        plan,
        scope_policy="strict_target",
        target_document_ids=["paper-1"],
        allow_supplemental=False,
        max_retrieval_rounds=2,
    )
    widened = plan.model_copy(deep=True)
    widened.retrieval_steps.append(RetrievalStep(tool="web_search", query="latest method"))

    enforced, audit = enforce_retrieval_contract(widened, contract)

    assert [step.tool for step in enforced.retrieval_steps] == ["hybrid_search"]
    assert audit.filtered_tools[0]["reason"] == "outside_contract"
    assert contract.target_document_ids == ["paper-1"]
    assert contract.allow_supplemental is False


def test_contract_allows_explicit_mixed_local_and_web_task():
    plan = _plan(
        RetrievalStep(tool="hybrid_search", query="knowledge base overview"),
        RetrievalStep(tool="web_search", query="latest progress"),
        required=["local_kb", "general_web"],
        preferred=["local_kb", "general_web"],
    )
    contract = compile_retrieval_contract(
        plan,
        scope_policy="broad_kb",
        target_document_ids=[],
        allow_supplemental=True,
        max_retrieval_rounds=2,
    )
    enforced, audit = enforce_retrieval_contract(plan, contract)

    assert [step.tool for step in enforced.retrieval_steps] == ["hybrid_search", "web_search"]
    assert audit.filtered_tools == []


def test_contract_marks_missing_required_evidence_as_insufficient():
    plan = _plan(
        RetrievalStep(tool="hybrid_search", query="local papers"),
        RetrievalStep(tool="openalex_search", query="related work"),
        required=["local_kb", "external_academic"],
        preferred=["local_kb", "external_academic"],
    )
    contract = compile_retrieval_contract(
        plan,
        scope_policy="broad_kb",
        target_document_ids=[],
        allow_supplemental=True,
        max_retrieval_rounds=2,
    )
    evaluation = evaluate_retrieval_contract(
        contract,
        results=[{"metadata": {"source_type": "local_kb"}, "content": "local evidence"}],
        tools_executed=[{"tool": "hybrid_search", "source_type": "local_kb", "result_count": 1}],
    )

    assert evaluation.required_sources_satisfied is False
    assert evaluation.missing_required_source_types == ["external_academic"]
    assert evaluation.reason == "missing_required_sources:external_academic"
