import pandas as pd


class RequestMetrics:
    COLUMN_TIMESTAMP = "timestamp"
    COLUMN_REQUEST_ID = "request_id"
    COLUMN_RESPONSE_CODE = "response_code"
    COLUMN_RESPONSE_PAYLOAD = "response_payload"
    COLUMN_MODEL_RESPONSE_CODE = "model_response_code"
    COLUMN_MODEL_RESPONSE_REASON = "model_response_reason"
    COLUMN_ADDITIONAL_WAIT_TIME = "additional_wait_time"
    COLUMN_REQUEST_TOTAL_WAIT_TIME = "request_total_wait_time"

    def __init__(self, metrics: pd.DataFrame = None) -> None:
        if metrics is not None:
            self.__validate_columns(metrics)
            self.__validate_index(metrics)
            self.__df = metrics.copy()
        else:
            self.__df = pd.DataFrame(columns=[
                RequestMetrics.COLUMN_TIMESTAMP,
                RequestMetrics.COLUMN_REQUEST_ID,
                RequestMetrics.COLUMN_RESPONSE_CODE,
                RequestMetrics.COLUMN_RESPONSE_PAYLOAD,
                RequestMetrics.COLUMN_MODEL_RESPONSE_CODE,
                RequestMetrics.COLUMN_MODEL_RESPONSE_REASON,
                RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME,
                RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME])
            self.__df.set_index(RequestMetrics.COLUMN_TIMESTAMP, inplace=True)

    def add_result(
        self,
        request_id: str,
        response_code: int,
        response_payload: any,
        model_response_code: str,
        model_response_reason: str,
        additional_wait_time: int,
        request_total_wait_time: int
    ):
        self.__df.loc[pd.Timestamp.utcnow()] = [
            request_id,
            response_code,
            response_payload,
            model_response_code,
            model_response_reason,
            additional_wait_time,
            request_total_wait_time]

    def get_metrics(self, start_time: pd.Timestamp, end_time: pd.Timestamp = None) -> pd.DataFrame:
        if end_time is None:
            end_time = pd.Timestamp.utcnow()

        # NOTE: This only works on desc sorted data. self.__df is sorted in desc by nature.
        return self.__df.loc[start_time:end_time]

    def __validate_columns(self, metrics: pd.DataFrame):
        expected_columns = [
            # RequestMetrics.COLUMN_TIMESTAMP is excluded, because it is the index.
            RequestMetrics.COLUMN_REQUEST_ID,
            RequestMetrics.COLUMN_RESPONSE_CODE,
            RequestMetrics.COLUMN_RESPONSE_PAYLOAD,
            RequestMetrics.COLUMN_MODEL_RESPONSE_CODE,
            RequestMetrics.COLUMN_MODEL_RESPONSE_REASON,
            RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME,
            RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME
        ]
        actual_columns = metrics.columns.tolist()

        if set(expected_columns) != set(actual_columns):
            raise ValueError(f"The metrics dataframe used to initialize RequestMetrics is invalid. "
                             f"Expected columns: {expected_columns}. "
                             f"Actual columns: {metrics.columns.tolist()}.")

    def __validate_index(self, metrics: pd.DataFrame):
        if metrics.index.name != RequestMetrics.COLUMN_TIMESTAMP:
            raise ValueError(f"The metrics dataframe used to initialize RequestMetrics is invalid. "
                             f"Expected index name: {RequestMetrics.COLUMN_TIMESTAMP}. "
                             f"Actual index name: {metrics.index.name}.")
