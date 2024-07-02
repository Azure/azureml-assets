# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image VQA predictor."""

import json
import re

from typing import List

from openai import AzureOpenAI

from logging_utilities import get_logger
from task_factory.base import PredictWrapper
from task_factory.image.constants import ImagePredictionsLiterals


ENDPOINT = ""
DEPLOYMENT = ""
API_VERSION = ""
API_KEY = ""


logger = get_logger(name=__name__)


class ImageVQAPredictor(PredictWrapper):
    """Predictor for visual question answering tasks.

    ???
    """

    def __init__(self, model_uri, task_type, device=None):
        """Initialize `ImageVQAPredictor` members."""
        # # Delegate to `PredictWrapper` constructor.
        super().__init__(model_uri, task_type, device)

        # self.answer_key = ImagePredictionsLiterals.ANSWER

        self.client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_version=API_VERSION,
            api_key=API_KEY,
        )

    def predict(self, X_test, **kwargs) -> List[str]:
        """???

        Args:
            X_test (pd.DataFrame): ???
        Returns:
            List[str]: ???
        """
        answers = []

        """
        # Get the batch size if specified, else set to default 1.
        batch_size = kwargs.get(ImagePredictionsLiterals.BATCH_SIZE, 1)

        for idx in range(0, len(X_test), batch_size):
            # Run model prediction on current batch.
            answer_batch = self.model.predict(X_test.iloc[idx:(idx + batch_size)])

            # Save batch of generated images in the default datastore.
            for answer in answer_batch[self.answer_key]:
                answers.append(answer)
        """
        print("d1", X_test.columns)

        for image, question, options, more_images in zip(
            X_test["image"], X_test["question"], X_test["answer_options"], X_test["more_images"]
        ):
            content = []
            previous = 0
            for match, current_image in zip(re.finditer("<image [0-9]>", question), [image] + more_images):
                i, j = match.span()
                if previous < i:
                    content.append({
                        "type": "text",
                        "text": question[previous:i]
                    })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{current_image}"
                    }
                })
                previous = j
            if previous < len(question):
                content.append({
                    "type": "text",
                    "text": question[previous:]
                })
            content.append({
                "type": "text",
                "text": "The options are" + "".join(
                    ["\n" + chr(ord("A") + i) + " " + option for i, option in enumerate(options.split("||"))]
                )
            })

            try:
                completion = self.client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": content,
                        },
                    ],
                )
                full_answer = json.loads(completion.to_json())
                answer = full_answer["choices"][0]["message"]["content"]
            except Exception as e:
                logger.info(f"question {question} produced exception {e}")
                answer = "Not available."

            answers.append(answer)

        return answers
