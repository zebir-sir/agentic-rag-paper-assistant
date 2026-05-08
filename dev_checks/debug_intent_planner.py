import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug Intent Planner and LangGraph retrieval-skip behavior.",
    )
    parser.add_argument("question", help="Question to debug.")
    parser.add_argument(
        "--context-hint",
        default="",
        help="Optional context hint passed into planner or graph.",
    )
    parser.add_argument(
        "--allow-openalex",
        action="store_true",
        help="Allow OpenAlex capability when provider is available.",
    )
    parser.add_argument(
        "--allow-web",
        action="store_true",
        help="Allow web capability when provider is available.",
    )
    parser.add_argument(
        "--allow-artifact",
        action="store_true",
        help="Accepted for explicit debugging intent; artifact search remains enabled by default.",
    )
    parser.add_argument(
        "--run-graph",
        action="store_true",
        help="Run the full LangGraph analysis flow instead of planner-only mode.",
    )
    return parser


def _extract_filtered_unavailable_tools(warnings: List[Any]) -> List[str]:
    filtered: List[str] = []
    for warning in warnings or []:
        text = str(warning or "").strip()
        prefix = "filtered_unavailable_tools:"
        if not text.startswith(prefix):
            continue
        items = [item.strip() for item in text[len(prefix):].split(",") if item.strip()]
        for item in items:
            if item not in filtered:
                filtered.append(item)
    return filtered


def _build_planner_capabilities_for_debug(args: argparse.Namespace) -> Any:
    from agent.intent_planner import PlannerCapabilities

    try:
        from agent.openalex_router import _is_openalex_enabled
    except Exception:
        provider_openalex = False
    else:
        provider_openalex = bool(_is_openalex_enabled())

    try:
        from agent.tools import is_general_web_search_enabled
    except Exception:
        provider_web = False
    else:
        provider_web = bool(is_general_web_search_enabled())

    return PlannerCapabilities(
        local_search_enabled=True,
        vector_search_enabled=True,
        hybrid_search_enabled=True,
        section_search_enabled=True,
        artifact_search_enabled=True,
        openalex_search_enabled=bool(args.allow_openalex and provider_openalex),
        web_search_enabled=bool(args.allow_web and provider_web),
        direct_answer_enabled=True,
        max_tools=2,
    )


def _build_debug_dependencies(args: argparse.Namespace) -> Any:
    from agent.agent_runtime import AgentDependencies

    return AgentDependencies(
        session_id="debug_intent_planner",
        user_id="dev_debug_user",
        use_web_search=bool(args.allow_web),
        search_preferences={
            "default_search_type": "hybrid",
            "default_limit": 10,
            "allow_web_search": bool(args.allow_web),
            "allow_openalex_search": bool(args.allow_openalex),
        },
    )


async def _run_planner_only(args: argparse.Namespace) -> Dict[str, Any]:
    from agent.intent_planner import plan_user_intent_debug

    capabilities = _build_planner_capabilities_for_debug(args)
    model_error = ""
    try:
        from agent.agent_langchain import get_langchain_chat_model

        model = get_langchain_chat_model()
    except Exception as exc:
        model = None
        model_error = str(exc)

    debug_payload = await plan_user_intent_debug(
        question=args.question,
        context_hint=args.context_hint,
        model=model,
        capabilities=capabilities,
    )
    plan = dict(debug_payload.get("normalized_plan") or {})
    planned_steps = list(plan.get("retrieval_steps") or [])
    warnings = list(plan.get("warnings") or [])
    direct_answer_allowed = bool(plan.get("direct_answer_allowed"))
    retrieval_skipped = bool((not plan.get("needs_retrieval")) and direct_answer_allowed)

    payload: Dict[str, Any] = {
        "mode": "planner_only",
        "question": args.question,
        "capabilities": dict(debug_payload.get("capabilities") or capabilities.model_dump()),
        "intent_plan_raw": debug_payload.get("raw_plan"),
        "raw_model_content_preview": str(debug_payload.get("raw_model_content_preview") or ""),
        "intent_plan": plan,
        "source_requirements": plan.get("source_requirements") or {},
        "answer_policy": plan.get("answer_policy") or {},
        "planned_retrieval_steps": planned_steps,
        "tools_planned": [str(step.get("tool") or "") for step in planned_steps if str(step.get("tool") or "").strip()],
        "tools_executed": [],
        "retrieval_skipped_by_planner": retrieval_skipped,
        "direct_answer_allowed": direct_answer_allowed,
        "filtered_unavailable_tools": _extract_filtered_unavailable_tools(warnings),
        "sources_count": 0,
        "warnings": warnings,
        "fallback_used": bool(debug_payload.get("fallback_used")),
        "fallback_reason": str(debug_payload.get("fallback_reason") or ""),
        "fallback_decision": str(debug_payload.get("fallback_decision") or ""),
    }
    if model_error:
        payload["planner_model_error"] = model_error
    return payload


async def _run_full_graph(args: argparse.Namespace) -> Dict[str, Any]:
    from agent.agent_langgraph import run_langgraph_analysis
    from agent.db_utils import close_database, initialize_database

    deps = _build_debug_dependencies(args)
    await initialize_database()
    try:
        result = await run_langgraph_analysis(
            question=args.question,
            deps=deps,
            context_prompt=args.context_hint or None,
        )
    finally:
        await close_database()

    metadata = dict((result.raw_state or {}).get("metadata") or result.metadata or {})
    payload: Dict[str, Any] = {
        "mode": "full_graph",
        "question": args.question,
        "planner_capabilities": metadata.get("planner_capabilities"),
        "intent": metadata.get("intent"),
        "intent_plan": metadata.get("intent_plan"),
        "source_requirements": metadata.get("source_requirements") or {},
        "answer_policy": metadata.get("answer_policy") or {},
        "fallback_used": bool(metadata.get("fallback_used")),
        "fallback_reason": str(metadata.get("fallback_reason") or ""),
        "fallback_decision": str(metadata.get("fallback_decision") or ""),
        "raw_model_content_preview": str(metadata.get("raw_model_content_preview") or ""),
        "planner_warnings": list(metadata.get("planner_warnings") or []),
        "retrieval_skipped_by_planner": bool(metadata.get("retrieval_skipped_by_planner")),
        "direct_answer_allowed": bool(metadata.get("direct_answer_allowed")),
        "planned_retrieval_steps": list(metadata.get("planned_retrieval_steps") or []),
        "tools_planned": list(metadata.get("tools_planned") or []),
        "tools_executed": list(metadata.get("tools_executed") or []),
        "filtered_unavailable_tools": list(metadata.get("filtered_unavailable_tools") or []),
        "sources_count": metadata.get("sources_count", metadata.get("source_count", 0)),
        "warnings": list(metadata.get("warnings") or list((result.raw_state or {}).get("warnings") or [])),
    }
    return payload


async def _async_main(args: argparse.Namespace) -> int:
    payload = await (_run_full_graph(args) if args.run_graph else _run_planner_only(args))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
