# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image VQA predictor."""

import json
import re

from typing import List

from openai import AzureOpenAI
from tqdm import tqdm

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
        # # Delegate to `PredictWrapper` constructor.
        super().__init__(model_uri, task_type, device)

        # self.answer_key = ImagePredictionsLiterals.ANSWER

        self.client, self.deployment = None, None

    def predict(self, X_test, **kwargs) -> List[str]:
        """???

        Args:
            X_test (pd.DataFrame): ???
        Returns:
            List[str]: ???
        """
        if (self.client is None) and (self.deployment is None):
            self.client = AzureOpenAI(
                azure_endpoint=kwargs["endpoint"],
                api_version=kwargs["api_version"],
                api_key=kwargs["api_key"],
            )
            self.deployment = kwargs["deployment"]

        answers = []

        for image, question, options, more_images in tqdm(zip(
            X_test["image"], X_test["question"], X_test["answer_options"], X_test["more_images"]
        )):
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
            if len(options) > 0:
                content.append({
                    "type": "text",
                    "text": "The options are" + "".join(
                        ["\n" + chr(ord("A") + i) + " " + option for i, option in enumerate(options.split("||"))]
                    )
                })

            try:
                completion = self.client.chat.completions.create(
                    model=self.deployment,
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
                if "429" in str(e):
                    raise
                logger.info(f"question {question} produced exception {e}")
                answer = "Not available."

            answers.append(answer)

        return answers
