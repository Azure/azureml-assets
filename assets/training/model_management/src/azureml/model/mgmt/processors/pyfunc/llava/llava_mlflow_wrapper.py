# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLFlow pyfunc wrapper for LLaVA models."""

import io
import os

import mlflow
import pandas as pd

from PIL import Image
from transformers import TextStreamer

from config import MAX_PROMPT_LENGTH, MLflowLiterals, MLflowSchemaLiterals, Tasks


class LLaVAMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for LLaVA models."""

    LLAVA_MPT = "mpt"
    LLAVA_7B = "7b"
    LLAVA_7B_15 = "7b_15"
    LLAVA_13B = "13b"
    LLAVA_13B2 = "13b2"
    LLAVA_13B_15 = "13b_15"
    MODEL_VERSIONS = [LLAVA_MPT, LLAVA_7B, LLAVA_7B_15, LLAVA_13B, LLAVA_13B2, LLAVA_13B_15]

    def __init__(self, task_type: str) -> None:
        """Construct LLaVA MLflow wrapper object.

        :param task_type: Task type for training or inference.
        :type task_type: str
        """
        super().__init__()

        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model

        if self._task_type == Tasks.IMAGE_TEXT_TO_TEXT.value:
            try:
                # Get the top level model directory (has main model directory + auxiliary directory in some cases).
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]

                # Infer the model version from the model directory.
                self._model_version = next(
                    (n for n in self.MODEL_VERSIONS if model_dir.endswith(n)), None
                )

                # Set the model name, model base and stop string from the model version.
                if self._model_version == self.LLAVA_MPT:
                    model_name = "LLaVA-Lightning-MPT-7B-preview"
                    model_base = None
                    stop_str = conv_templates["mpt"].sep

                elif self._model_version == self.LLAVA_7B:
                    model_name = "llava-llama-2-7b-chat-lightning-lora-preview"
                    model_base = os.path.join(model_dir, "Llama-2-7b-chat")
                    stop_str = conv_templates["llava_llama_2"].sep

                elif self._model_version == self.LLAVA_7B_15:
                    model_name = "llava-v1.5-7b"
                    model_base = None
                    stop_str = conv_templates["llava_v1"].sep2

                elif self._model_version == self.LLAVA_13B:
                    model_name = "llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
                    model_base = None
                    stop_str = conv_templates["llava_v1"].sep2

                elif self._model_version == self.LLAVA_13B2:
                    model_name = "llava-llama-2-13b-chat-lightning-preview"
                    model_base = None
                    stop_str = conv_templates["llava_llama_2"].sep

                elif self._model_version == self.LLAVA_13B_15:
                    model_name = "llava-v1.5-13b"
                    model_base = None
                    stop_str = conv_templates["llava_v1"].sep2

                else:
                    raise NotImplementedError(f"Model name {self._model_version} not supported.")

                # Make the model and preprocessing objects.
                self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
                    os.path.join(model_dir, model_name), model_base, model_name, False, False
                )
                self._streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
                self._stop_str = stop_str

                print("Model loaded successfully")

            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise

        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with columns ["image"], ["prompt"] and ["direct_question"], where
                           the image is either a url or a base64 string, the prompt is the dialog so far between the
                           user and the model and the direct question is a prompt with a single question from the user.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with column ["response"] containing the model's response to the dialog so far.
        """
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from vision_utils import process_image

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
                prompt_from_direct_question = True
                if self._model_version == self.LLAVA_MPT:
                    prompt = (
                        f"A conversation between a user and an LLM-based AI assistant. The assistant gives helpful "
                        f"and honest answers.<|im_end|><|im_start|>user\n<im_start><image><im_end>\n"
                        f"{direct_question}<|im_end|><|im_start|>assistant"
                    )
                elif self._model_version == self.LLAVA_7B:
                    prompt = (
                        f"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand "
                        f"the visual content that the user provides, and assist the user with a variety of tasks "
                        f"using natural language.\n<</SYS>>\n\n<im_start><image><im_end>\n{direct_question} [/INST]"
                    )
                elif self._model_version == self.LLAVA_7B_15:
                    prompt = (
                        f"A chat between a curious human and an artificial intelligence assistant. "
                        f"The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
                        f"<image>\n{direct_question} ASSISTANT:"
                    )
                elif self._model_version == self.LLAVA_13B:
                    prompt = (
                        f"A chat between a curious human and an artificial intelligence assistant. "
                        f"The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
                        f"<im_start><image><im_end>\n{direct_question} ASSISTANT:"
                    )
                elif self._model_version == self.LLAVA_13B2:
                    prompt = (
                        f"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand "
                        f"the visual content that the user provides, and assist the user with a variety of tasks "
                        f"using natural language.\n<</SYS>>\n\n<image>\n{direct_question} [/INST]"
                    )
                elif self._model_version == self.LLAVA_13B_15:
                    prompt = (
                        f"A chat between a curious human and an artificial intelligence assistant. "
                        f"The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
                        f"<image>\n{direct_question} ASSISTANT:"
                    )
            else:
                prompt_from_direct_question = False

            # Make image input.
            image_tensor = self._image_processor.preprocess(
                pil_image, return_tensors="pt"
            )["pixel_values"].half().cuda()

            # Make text input.
            input_ids = tokenizer_image_token(
                prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria([self._stop_str], self._tokenizer, input_ids)

            # For small models on V100 machines, long prompts cause a GPU OOMs which the server does not recover from.
            # To prevent this, we are using a length threshold that allows for a small number of question-answer pairs
            # (e.g. 5-10) in each prompt.
            if self._model_version in [self.LLAVA_MPT, self.LLAVA_7B, self.LLAVA_7B_15]:
                prompt_length = max([len(i) for i in input_ids])
                if prompt_length > MAX_PROMPT_LENGTH:
                    raise ValueError(
                        f"Prompt too long: {prompt_length} tokens. Maximum allowed is {MAX_PROMPT_LENGTH}."
                    )

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
            if prompt_from_direct_question:
                if response.startswith(": "):
                    response = response[2:]
                if response.endswith(self._stop_str):
                    response = response[:-len(self._stop_str)]
                if response.endswith("</s>"):
                    response = response[:-len("</s>")]

            # Accumulate into response list.
            responses.append(response)

        # Convert responses to Pandas dataframe.
        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE: responses})
        return df_responses
