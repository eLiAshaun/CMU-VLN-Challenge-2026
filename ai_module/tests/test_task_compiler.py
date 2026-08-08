import integrations.semantics.task_compiler as compiler
from integrations.semantics.task_compiler import compile_task, deterministic_compile, validate_task_ir


QUESTIONS = [
    "How many photos are on the TV cabinet?",
    "Find the potted plant near the books on the cabinet.",
    "Find the vase between the cabinet and the stool.",
    "Take the path near the TV and go to the pillow farthest from the lamp.",
    "First, go near the stool, then take the path near the cabinet, and stop at the bowl on the table.",
]


def local(question):
    return validate_task_ir(deterministic_compile(question), question)


def test_livingroom3_count_preserves_target_and_support():
    task = local(QUESTIONS[0])
    assert task["task_type"] == "numerical"
    assert task["entities"][0]["class_name"] == "picture"
    assert task["entities"][1]["class_name"] == "television cabinet"
    assert task["relations"][0]["predicate"] == "on"
    assert task["output_contract"] == "/numerical_response"


def test_livingroom3_nested_and_between_relations_are_typed():
    nested = local(QUESTIONS[1])
    assert [item["predicate"] for item in nested["relations"]] == ["on", "near"]
    assert nested["required_classes"] == ["potted plant", "book", "cabinet"]
    between = local(QUESTIONS[2])
    assert between["relations"][0]["predicate"] == "between"
    assert between["required_classes"] == ["vase", "cabinet", "stool"]


def test_livingroom3_instruction_order_and_qualifiers_are_preserved():
    first = local(QUESTIONS[3])
    assert [item["action"] for item in first["ordered_subgoals"]] == ["pass_near", "go_to"]
    assert first["relations"][0]["predicate"] == "farthest"
    assert first["required_classes"] == ["television", "pillow", "lamp"]
    second = local(QUESTIONS[4])
    assert [item["action"] for item in second["ordered_subgoals"]] == ["go_near", "pass_near", "stop_at"]
    assert second["relations"][0]["predicate"] == "on"
    assert "bowl" in second["required_classes"]


def test_missing_deepseek_credential_is_explicit_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    task = compile_task(QUESTIONS[0], {"enabled": True})
    diagnostics = task["compiler_diagnostics"]
    assert diagnostics["backend"] == "deterministic_v1"
    assert diagnostics["deepseek_called"] is True
    assert "credential_unavailable" in diagnostics["fallback_reason"]


def test_deepseek_can_compile_language_outside_local_templates(monkeypatch):
    question = "Could you point out whichever vase sits between the cabinet and stool?"
    remote = local("Find the vase between the cabinet and the stool.")
    remote["original_question"] = question
    monkeypatch.setattr(compiler, "_deepseek_compile", lambda *_args, **_kwargs: remote)
    task = compile_task(question, {"enabled": True})
    assert task["compiler_diagnostics"]["backend"] == "deepseek"
    assert task["compiler_diagnostics"]["local_parser_ready"] is False
    assert task["required_classes"] == ["vase", "cabinet", "stool"]
