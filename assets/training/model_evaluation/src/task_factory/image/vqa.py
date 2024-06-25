# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image VQA predictor."""

from typing import List

from logging_utilities import get_logger
from task_factory.base import PredictWrapper
from task_factory.image.constants import ImagePredictionsLiterals


logger = get_logger(name=__name__)


class ImageVQAPredictor(PredictWrapper):
    """Predictor for visual question answering tasks.

    ???
    """

    def __init__(self, model_uri, task_type, device=None):
        """Initialize `ImageVQAPredictor` members."""
        # Delegate to `PredictWrapper` constructor.
        super().__init__(model_uri, task_type, device)

        # ???
        self.answer_key = ImagePredictionsLiterals.ANSWER

    def predict(self, X_test, **kwargs) -> List[str]:
        """???

        Args:
            X_test (pd.DataFrame): ???
        Returns:
            List[str]: ???
        """
        # Get the batch size if specified, else set to default 1.
        batch_size = kwargs.get(ImagePredictionsLiterals.BATCH_SIZE, 1)

        answers = []

        for idx in range(0, len(X_test), batch_size):
            # Run model prediction on current batch.
            answer_batch = self.model.predict(X_test.iloc[idx:(idx + batch_size)])

            # Save batch of generated images in the default datastore.
            for answer in answer_batch[self.answer_key]:
                answers.append(answer)

        return answers
