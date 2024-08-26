from promptflow import tool
from constants import Service, ServiceVersion, MetricGroup


def normalize_user_text(user_text):
    return user_text.replace("'", "\\\"")


@tool
def construct_request(question: str,
                      answer: str,
                      selected_metrics: dict) -> dict:
    selected_safety_metrics = selected_metrics[MetricGroup.SAFETY_METRICS]
    metrics = [metric.replace("_unfairness", "_fairness") for metric in
               selected_safety_metrics if selected_safety_metrics[metric]]
    user_text = f"<Human>{question}</><System>{answer}</>"
    parsed_user_text = normalize_user_text(user_text)
    prompt_version = ServiceVersion.ContentHarm
    request_body = {"UserTextList": [parsed_user_text],
                    "AnnotationTask": Service.ContentHarm,
                    "MetricList": metrics,
                    "PromptVersion": prompt_version}
    return request_body
