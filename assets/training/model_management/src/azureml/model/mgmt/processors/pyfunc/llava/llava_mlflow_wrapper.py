# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
MLFlow pyfunc wrapper for LLaVA models.
"""

import io
import os

import mlflow
import pandas as pd

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image
from transformers import TextStreamer

from constants import MLflowLiterals, MLflowSchemaLiterals
from utils import process_image


class LLaVAMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for LLaVA models."""

    LLAVA_MPT = "LLaVA-Lightning-MPT-7B-preview"
    LLAVA_7B = "llava-llama-2-7b-chat-lightning-lora-preview"
    LLAVA_13B = "llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
    MODEL_NAMES = [LLAVA_MPT, LLAVA_7B, LLAVA_13B]

    def __init__(self, task_type: str) -> None:
        """Constructor for MLflow wrapper class
        :param task_type: Task type for training or inference.
        :type task_type: str
        """
        super().__init__()

        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().
        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """

        try:
            # Get the model directory.
            model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]

            # Infer the model name from the model directory.
            self._model_name = next((n for n in self.MODEL_NAMES if n in model_dir), None)

            # Set the model base and stop string from the model name.
            if self._model_name == self.LLAVA_MPT:
                model_base = None
                stop_str = conv_templates["mpt"].sep

            elif self._model_name == self.LLAVA_7B:
                model_base = os.path.join(model_dir, "Llama-2-7b-chat")
                stop_str = conv_templates["llava_llama_2"].sep

            elif self._model_name == self.LLAVA_13B:
                model_base = None
                stop_str = conv_templates["llava_v1"].sep2

            else:
                raise NotImplementedError(f"Model name {self._model_name} not supported.")

            # Make the model and preprocessing objects.
            self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(model_dir + "/", model_base, self._model_name, False, False)
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
        :param input_data: Pandas DataFrame with columns ["image"], ["prompt"] and ["direct_question"], where
                           the image is either a url or a base64 string, the prompt is the dialog so far between the
                           user and the model and the direct question is a prompt with a single question from the user.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with column ["response"] containing the model's response to the dialog so far.
        """

        # Do inference one input at a time.
        responses = []
        for image, prompt, direct_question in zip(
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_PROMPT],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_DIRECT_QUESTION],
        ):
            # Decode the image and make a PIL Image object.
            pil_image = Image.open(io.BytesIO(process_image(image)))

            # If prompt not specified, make prompt from direct question column.
            if not prompt:
                if self._model_name == self.LLAVA_MPT:
                    prompt = f"A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n<im_start><image><im_end>\n{direct_question}<|im_end|><|im_start|>assistant"
                elif self._model_name == self.LLAVA_7B:
                    prompt = f"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<im_start><image><im_end>\n{direct_question} [/INST]"
                elif self._model_name == self.LLAVA_13B:
                    prompt = f"input to llava: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\n{direct_question} ASSISTANT:"

            # Make image input.
            image_tensor = self._image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"].half().cuda()

            # Make text input.
            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria([self._stop_str], self._tokenizer, input_ids)

            # Call model.
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

            # Convert response to text and trim.
            response = self._tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if response.startswith(": "):
                response = response[2:]
            if response.endswith(self._stop_str):
                response = response[:-len(self._stop_str)]

            # Accumulate into response list.
            responses.append(response)

        # Convert responses to Pandas dataframe.
        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE: responses})
        return df_responses
