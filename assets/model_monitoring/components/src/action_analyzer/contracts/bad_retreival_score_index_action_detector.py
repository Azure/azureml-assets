# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bad retrieval score index action detector class."""

from action_analyzer.contracts.action_detector import ActionDetector
from action_analyzer.contracts.action import Action


class BadRetreivalScoreIndexActionDetector(ActionDetector):
    """Bad retrieval score index action detector class."""

    def __init__(self, 
                 workspace_connection_arm_id: str,
                 model_deployment_name: str,
                 aml_deployment_id: str,
                 retrieval_score_violation_threshold: float,
                 retrieval_score_violation_rate: float,
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str):
        """Create a bad retrieval score index action detector.

        Args:
            workspace_connection_arm_id(str): azureml workspace connection arm id for llm.
            model_deployment_name(str): model deployment name of the connection.
            aml_deployment_id(str): the azureml deployment id of the llm application.
            retrieval_score_violation_threshold(float): threshold for retrieval score violation.
            retrieval_score_violation_rate(float): retrieval score violation rate. If the violation rate is below this number, an action will be generated.
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm summary. Accepted values: true or false.
        """
        self.retrieval_score_violation_threshold = retrieval_score_violation_threshold
        self.retrieval_score_violation_rate = retrieval_score_violation_rate
        super().__init__(workspace_connection_arm_id, model_deployment_name, aml_deployment_id, action_max_positive_sample_size, llm_summary_enabled)


    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        pass


    def detect(self, df) -> list(Action):
        """Detect the action.
        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            list(Action): list of actions.
        """
        pass
