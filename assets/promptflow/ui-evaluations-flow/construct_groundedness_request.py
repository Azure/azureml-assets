from promptflow import tool
import json
from constants import Service


def normalize_user_text(user_text):
    return user_text.replace("'", "\\\"")


@tool
def construct_request(answer: str,
                      context: str,
                      question: str = "") -> dict:
    metrics = ["generic_groundedness"]
    user_text = json.dumps({"question": question,
                            "answer": answer,
                            "context": context})
    parsed_user_text = normalize_user_text(user_text)
    request_body = {"UserTextList": [parsed_user_text],
                    "AnnotationTask": Service.Groundedness,
                    "MetricList": metrics}
    return request_body
