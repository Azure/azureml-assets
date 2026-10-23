# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Record CI-built candidate wheel provenance and require complete native test results."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


MINIMUM_CASES = {
    "test_openai_decisions": 76,
    "test_evaluators": 45, "test_composites": 274,
    "test_quality_expansion": 132, "test_tool_expansion": 92, "test_judge_expansion": 108,
    "test_expansion_compatibility": 60,
    "test_output_quality_evaluator_behavior": 115, "test_tool_use_quality_evaluator_behavior": 132,
}
REQUIRED_MODULE_CASES = {
    "test_output_quality_evaluator_behavior": {
        f"test_native_decision_one_transport_twelve_questions_and_no_chat[{mode}]"
        for mode in ("sync", "async", "sync_in_loop")
    },
    "test_tool_use_quality_evaluator_behavior": {
        f"test_native_decision_one_request_ten_questions_and_metadata[{shape}-{level}]"
        for shape in ("query", "messages") for level in ("None", "turn", "conversation")
    },
    "test_composites": {
        *(f"test_direct_task_cancellation_retains_prior_paid_native_usage[{name}]"
          for name in ("TaskCompletionEvaluator", "OutputQualityEvaluator", "ToolUseQualityEvaluator")),
        *(f"test_output_quality_queryless_conversation_keeps_response[{response}-{query}-{backend}]"
          for response in ("text", "list")
          for query in ("omitted", "none", "empty_text", "empty_list")
          for backend in ("native", "chat")),
    },
    "test_quality_expansion": {
        *(f"test_conversation_public_calls_preserve_both_input_sides[{shape}-{backend}-{name}]"
          for shape in ("text_list", "list_text", "text_text", "list_list", "messages")
          for backend in ("native", "chat")
          for name in ("CoherenceEvaluator", "GroundednessEvaluator", "TaskAdherenceEvaluator",
                       "CustomerSatisfactionEvaluator")),
        *(f"test_groundedness_queryless_conversation_preserves_response_and_context[{response}-{query}-{backend}]"
          for response in ("text", "list")
          for query in ("omitted", "none", "empty_text", "empty_list")
          for backend in ("native", "chat")),
        *(f"test_groundedness_explicit_assistant_only_messages_remain_invalid[{backend}]"
          for backend in ("native", "chat")),
        *(f"test_groundedness_queryless_conversation_requires_original_context[{context}-{response}-{query}-{backend}]"
          for context in ("omitted", "none", "empty_text", "empty_list")
          for response in ("text", "list")
          for query in ("omitted", "none", "empty_text", "empty_list")
          for backend in ("native", "chat")),
    },
}


def report_module(case):
    """Resolve the module component for both function and class-based pytest cases."""
    classname = case.get("classname", "")
    modules = set(classname.split(".")) & MINIMUM_CASES.keys()
    assert len(modules) == 1, f"Unexpected or ambiguous native classname: {classname}"
    return modules.pop()


REQUIRED_CASES = {
    "test_completion_sync_without_prompty", "test_accuracy_score_and_metadata",
    *(f"test_public_evaluate_preserves_member_provenance_and_root_usage[passed-{name}]"
      for name in ("OutputQualityEvaluator", "ToolUseQualityEvaluator")),
    *(f"test_packaged_expansion_public_usage[{skipped}-{backend}-{name}]"
      for name in (
          "fluency", "coherence", "groundedness", "intent_resolution", "task_adherence",
          "tool_call_success", "tool_input_accuracy", "tool_output_utilization", "tool_selection",
          "customer_satisfaction", "deflection_rate", "relevance", "response_completeness", "retrieval", "similarity",
      )
      for backend in ("native", "chat") for skipped in ("False", "True")),
}


def verify_report(report):
    """Allow additional cases but fail missing native coverage or any skips/errors/failures."""
    tree = ET.parse(report)
    cases = tree.findall(".//testcase")
    owned_cases = [(report_module(case), case) for case in cases]
    counts = Counter(module for module, _ in owned_cases)
    assert set(counts) == set(MINIMUM_CASES), f"Unexpected native modules: {counts}"
    assert all(counts[name] >= minimum for name, minimum in MINIMUM_CASES.items()), counts
    names = {case.get("name") for case in cases}
    assert REQUIRED_CASES <= names, f"Missing native cases: {REQUIRED_CASES - names}"
    for module, required in REQUIRED_MODULE_CASES.items():
        names = {case.get("name") for owner, case in owned_cases if owner == module}
        assert required <= names, f"Missing native cases in {module}: {required - names}"
    assert not any(tree.findall(f".//{tag}") for tag in ("skipped", "failure", "error")), "Native cases must pass"
    print(f"Verified native cases: {dict(counts)}")


def main():
    """Verify the installed candidate and retain its generated, not preapproved, wheel hash."""
    import azure.ai.evaluation as sdk
    from azure.ai.evaluation._version import VERSION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--provenance-only", action="store_true")
    args = parser.parse_args()
    installed = distribution("azure-ai-evaluation")
    assert installed.version == VERSION == os.environ["SDK_CANDIDATE_VERSION"]
    assert Path(sdk.__file__).resolve().is_relative_to(Path(installed.locate_file("")).resolve()), sdk.__file__
    provenance = {
        "source_repository": "https://github.com/Azure/azure-sdk-for-python.git",
        "source_commit": os.environ["SDK_CANDIDATE_COMMIT"], "wheel_version": VERSION,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "candidate-native-sdk-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(provenance)
    if not args.provenance_only:
        verify_report(args.reports / "candidate-native-sdk.xml")


if __name__ == "__main__":
    main()
