import copy
from typing import Any, Dict, List, Optional, Tuple


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "hybrid_search": {
        "type": "function",
        "name": "hybrid_search",
        "description": "Search the local knowledge base using hybrid retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "source_type": "local_kb",
        "availability_key": "hybrid_search_enabled",
    },
    "vector_search": {
        "type": "function",
        "name": "vector_search",
        "description": "Search the local knowledge base using vector similarity only.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "source_type": "local_kb",
        "availability_key": "vector_search_enabled",
    },
    "section_search": {
        "type": "function",
        "name": "section_search",
        "description": "Search section-scoped evidence from local documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "section_query": {"type": "string"},
                "document_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "source_type": "local_section",
        "availability_key": "section_search_enabled",
    },
    "artifact_search": {
        "type": "function",
        "name": "artifact_search",
        "description": "Search local tables, figures, and algorithms.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "artifact_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["table", "figure", "algorithm"],
                    },
                },
                "document_id": {"type": "string"},
            },
            "required": ["query"],
        },
        "source_type": "local_artifact",
        "availability_key": "artifact_search_enabled",
    },
    "openalex_search": {
        "type": "function",
        "name": "openalex_search",
        "description": "Search external academic paper metadata through OpenAlex.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        "source_type": "external_academic",
        "availability_key": "openalex_search_enabled",
    },
    "web_search": {
        "type": "function",
        "name": "web_search",
        "description": "Search general web sources for current or online information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        "source_type": "general_web",
        "availability_key": "web_search_enabled",
    },
    "get_document": {
        "type": "function",
        "name": "get_document",
        "description": "Fetch a local document by document_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
            },
            "required": ["document_id"],
        },
        "source_type": "local_kb",
        "availability_key": "local_search_enabled",
    },
    "list_documents": {
        "type": "function",
        "name": "list_documents",
        "description": "List documents currently available in the local knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000},
            },
            "required": [],
        },
        "source_type": "local_kb",
        "availability_key": "local_search_enabled",
    },
}


def get_tool_spec(name: str) -> Optional[Dict[str, Any]]:
    spec = TOOL_SPECS.get(str(name or "").strip())
    return copy.deepcopy(spec) if spec is not None else None


def get_tool_source_type(name: str) -> Optional[str]:
    spec = TOOL_SPECS.get(str(name or "").strip())
    if spec is None:
        return None
    return str(spec.get("source_type") or "").strip() or None


def _tool_enabled(spec: Dict[str, Any], capabilities: Any) -> bool:
    if capabilities is None:
        return True
    key = str(spec.get("availability_key") or "").strip()
    if not key:
        return True
    return bool(getattr(capabilities, key, False))


def get_openai_tool_specs(capabilities: Any = None) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for name in TOOL_SPECS:
        spec = TOOL_SPECS[name]
        if not _tool_enabled(spec, capabilities):
            continue
        specs.append(copy.deepcopy(spec))
    return specs


def _validate_scalar(expected_type: str, value: Any) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def validate_tool_arguments(name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
    spec = TOOL_SPECS.get(str(name or "").strip())
    if spec is None:
        return False, "unknown_tool"
    if not isinstance(arguments, dict):
        return False, "arguments_must_be_object"

    schema = dict(spec.get("parameters") or {})
    if schema.get("type") != "object":
        return False, "invalid_schema"

    properties = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])

    for key in required:
        value = arguments.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"missing_required:{key}"

    for key, value in arguments.items():
        if key not in properties:
            return False, f"unknown_argument:{key}"
        prop = dict(properties.get(key) or {})
        expected_type = str(prop.get("type") or "").strip()
        if expected_type and not _validate_scalar(expected_type, value):
            return False, f"invalid_type:{key}"
        if expected_type == "string" and isinstance(value, str) and not value.strip() and key in required:
            return False, f"empty_string:{key}"
        if expected_type in {"integer", "number"} and value is not None:
            minimum = prop.get("minimum")
            maximum = prop.get("maximum")
            if minimum is not None and value < minimum:
                return False, f"below_minimum:{key}"
            if maximum is not None and value > maximum:
                return False, f"above_maximum:{key}"
        if expected_type == "array":
            if not isinstance(value, list):
                return False, f"invalid_type:{key}"
            item_schema = dict(prop.get("items") or {})
            item_type = str(item_schema.get("type") or "").strip()
            allowed_values = set(item_schema.get("enum") or [])
            for item in value:
                if item_type and not _validate_scalar(item_type, item):
                    return False, f"invalid_array_item:{key}"
                if allowed_values and item not in allowed_values:
                    return False, f"invalid_enum:{key}"

    return True, ""
