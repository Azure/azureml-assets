# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
MLFlow pyfunc wrapper for LLaVA models.
"""

import base64
import io
import os
import re
import requests
import subprocess

# os.system("git clone https://github.com/haotian-liu/LLaVA.git && cd LLaVA && pip install -e .")
# print(subprocess.check_output(["git", "clone", "https://github.com/haotian-liu/LLaVA.git"], shell=True))
# print(subprocess.check_output(["cd", "LLaVA"], shell=True))
# print(__file__)
# print(os.path.dirname(__file__))
# print(os.path.abspath(os.path.dirname(__file__)))
# print(subprocess.check_output(["ls", os.path.dirname(__file__)], shell=True))
# os.system(f"pip install --user {os.path.dirname(__file__)}/LLaVA")

import mlflow
import pandas as pd

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image
from transformers import TextStreamer


class MLflowLiterals:
    """
    MLflow export related literals
    """
    MODEL_DIR = "model_dir"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_PROMPT = "prompt"

    OUTPUT_COLUMN_RESPONSE = "response"


def _process_image(img: pd.Series) -> pd.Series:
    """Process input image.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url or bytes.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if isinstance(image, bytes):
        return img
    elif isinstance(image, str):
        if _is_valid_url(image):
        # if True:
            image = requests.get(image).content
            return pd.Series(image)
        else:
            try:
                return pd.Series(base64.b64decode(image))
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded. Expected format is Base64 or URL string."
                )
    else:
        raise ValueError(
            f"Image received in {type(image)} format which is not supported."
            "Expected format is bytes, base64 string or url string."
        )


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=\\-]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


class LLaVAMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for LLaVA models."""

    def __init__(
            self,
            task_type: str,
            model_id: str,
    ) -> None:
        """Constructor for MLflow wrapper class
        :param task_type: Task type used in training.
        :type task_type: str
        :param model_id: Hugging face model id corresponding to LLaVA models supported by AML.
        :type model_id: str
        """
        super().__init__()

        self._task_type = task_type
        self._model_id = model_id

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().
        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """

        print("x1")

        # Install llava package from source.
        # subprocess.call([
        #     "git", "clone", "https://github.com/haotian-liu/LLaVA.git", " && ",
        #     "cd", "LLaVA", " && ",
        #     "pip", "install", "-e", "."
        # ])
        # os.system("git clone https://github.com/haotian-liu/LLaVA.git && cd LLaVA && pip install -e .")

        try:
            # _map_location = "cuda" if torch.cuda.is_available() else "cpu"

            model_dir = context.artifacts[MLflowLiterals.MODEL_DIR] + "/"

            self._model_id = "mpt"
            if self._model_id == "mpt":
                model_base = None
                model_name = "LLaVA-Lightning-MPT-7B-preview"
                stop_str = conv_templates["mpt"].sep

            elif self._model_id == "7b":
                model_name = "llava-llama-2-7b-chat-lightning-lora-preview"
                model_base = context.artifacts[MLflowLiterals.MODEL_DIR].replace(model_name, "Llama-2-7b-chat")
                stop_str = conv_templates["llava_llama_2"].sep

            elif self._model_id == "13b":
                model_base = None
                model_name = "llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
                stop_str = conv_templates["mpt"].sep2

            else:
                model_base, model_name, stop_str = None, None, None

            print("x2")
            self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(model_dir, model_base, model_name, False, False)
            self._streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            self._stop_str = stop_str

            print("Model loaded successfully")

        except Exception as e:
            print("Failed to load the the model.")
            print(e)
            raise

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform inference on the input data.
        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with a column name ["prompt"] having text
                           input for which image has to be generated.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input text prompts, their corresponding generated images and NSFW flag.
                 Images in form of base64 string.

        """

        # Preprocess the images.
        decoded_images = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]].apply(axis=1, func=_process_image)

        # Make the list of prompts.
        prompts = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_PROMPT]]

        print("y1", len(decoded_images), len(prompts))

        # Do inference.
        responses = []
        for (_, image), (__, prompt) in zip(decoded_images.iterrows(), prompts.iterrows()):
            print("y2")

            decoded_image = Image.open(io.BytesIO(image[0]))
            prompt = prompt[0]
            # prompt = f"A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n<im_start><image><im_end>\n{prompt[0]}<|im_end|><|im_start|>assistant"

            print("y3")

            image_tensor = self._image_processor.preprocess(decoded_image, return_tensors="pt")["pixel_values"].half().cuda()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria([self._stop_str], self._tokenizer, input_ids)

            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=self._streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

            response = self._tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if response.startswith(": "):
                response = response[2:]
            if response.endswith("<|im_end|>"):
                response = response[:-10]
            responses.append(response)

        # Convert responses to Pandas dataframe.
        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE: responses})
        return df_responses
