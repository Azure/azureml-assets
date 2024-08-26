from promptflow import tool
import mlflow
from mlflow.utils.rest_utils import http_request
from utils import get_cred
from promptflow._utils.logger_utils import logger
from constants import Service, MetricGroup, Metric


def is_service_available(flight: bool):
    content_harm_service = False
    groundedness_service = False
    try:
        cred = get_cred()

        response = http_request(
            host_creds=cred,
            endpoint="/checkannotation",
            method="GET",
        )

        if response.status_code != 200:
            logger.error(
                "Failed to get RAI service availability in this region." +
                "Response_code: %d" % response.status_code
                )
        else:
            available_service = response.json()
            # check if content harm service is avilable
            if Service.ContentHarm in available_service:
                content_harm_service = True
            else:
                logger.warning(
                    "RAI service is not available in this region. " +
                    "Fallback to template-based groundedness evaluation."
                    )
            # check if groundedness service is avilable
            if Service.Groundedness in available_service and flight:
                groundedness_service = True
            if not flight:
                logger.warning(
                    "GroundednessServiceFlight is off. " +
                    "Fallback to template-based groundedness evaluation."
                    )
            if Service.Groundedness not in available_service:
                logger.warning(
                    "Groundedness service is not available in this region. " +
                    "Fallback to template-based groundedness evaluation."
                    )
    except Exception:
        logger.error("Failed to call checkannotation endpoint.")
    return {"content_harm_service": content_harm_service,
            "groundedness_service": groundedness_service
            }


def is_tracking_uri_set():
    if not mlflow.is_tracking_uri_set():
        logger.error("tracking_uri is not set")
        return False
    else:
        return True


def is_safety_metric_selected(selected_metrics: dict) -> bool:
    selected_safety_metrics = selected_metrics[MetricGroup.SAFETY_METRICS]
    for metric in selected_safety_metrics:
        if selected_safety_metrics[metric]:
            return True
    logger.info("No safety metrics are selected.")
    return False


def is_groundedness_metric_selected(selected_metrics: dict) -> bool:
    groundedness_selected = selected_metrics[
        MetricGroup.QUALITY_METRICS][Metric.GPTGroundedness]
    if not groundedness_selected:
        logger.info("%s is not selected." % Metric.GPTGroundedness)
    return groundedness_selected


# check if RAI service is avilable in this region. If not, return False.
# check if tracking_uri is set. If not, return False
# if tracking_rui is set, check if any safety metric is selected.
# if no safety metric is selected, return False
@tool
def validate_safety_metric_input(
        selected_metrics: dict,
        validated_input: dict,
        flight: str = "true",
        ) -> dict:
    tracking_uri_set = is_tracking_uri_set()
    flight_bool = False if "false" in flight.lower() else True
    service_available = is_service_available(flight_bool)
    safety_metrics_selected = is_safety_metric_selected(selected_metrics)
    gpt_groundedness_selected = is_groundedness_metric_selected(
        selected_metrics)
    valid_input_groundedness = validated_input[Metric.GPTGroundedness]
    valid_input_safety = validated_input[MetricGroup.SAFETY_METRICS]

    content_harm_service = safety_metrics_selected \
        and valid_input_safety and tracking_uri_set \
        and service_available["content_harm_service"]

    groundedness_service = gpt_groundedness_selected\
        and valid_input_groundedness\
        and tracking_uri_set \
        and service_available["groundedness_service"]

    groundedness_prompt = gpt_groundedness_selected \
        and valid_input_groundedness \
        and (not service_available["groundedness_service"])

    if not valid_input_groundedness and gpt_groundedness_selected:
        logger.error("Input is not valid for %s evaluation"
                     % Metric.GPTGroundedness)

    if not valid_input_safety and safety_metrics_selected:
        logger.error("Input is not valid for %s evaluation"
                     % MetricGroup.SAFETY_METRICS)

    return {"content_harm_service": content_harm_service,
            "groundedness_service": groundedness_service,
            "groundedness_prompt": groundedness_prompt
            }
