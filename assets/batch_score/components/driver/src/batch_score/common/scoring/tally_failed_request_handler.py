from ..telemetry import logging_utils as lu


class TallyFailedRequestHandler(object):
    def __init__(self, enabled: bool, tally_exclusions: str = None):
        self.__enabled = enabled
        self.__exclusions: list[str] = []

        if self.__enabled and tally_exclusions:
            self.__exclusions = [exclusion.strip().lower() for exclusion in tally_exclusions.split('|')]

            if "none" in self.__exclusions and len(self.__exclusions) > 1:
                raise Exception("Conflicting tally_exclusions: \"none\" specified alongside other exclusions.")

    def should_tally(self, response_status: int, model_response_status: int) -> bool:
        if not self.__enabled:
            return False

        failure_category = TallyFailedRequestHandler._categorize(
            response_status=response_status,
            model_response_status=model_response_status)

        should_tally = failure_category not in self.__exclusions

        lu.get_logger().debug(f"should_tally: {should_tally}")
        return should_tally

    @staticmethod
    def _categorize(response_status: int, model_response_status: int):
        category: str = None
        if response_status == 424 and model_response_status == 400:
            category = "bad_request_to_model"

        lu.get_logger().debug(f"Failed request categorized as: {category}")
        return category.lower() if category else None
