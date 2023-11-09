# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for segment anything models.

Has methods to load the model and predict.
"""


import io
import mlflow
import pandas as pd
from PIL import Image
import torch
from transformers import SamModel, SamProcessor

from config import MLflowSchemaLiterals, Tasks, MLflowLiterals, SAMHFLiterals, DatatypeLiterals
from vision_utils import process_image, string_to_nested_float_list, image_to_base64, bool_array_to_pil_image


class SegmentAnythingMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for segment anything models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize model parameters for converting Huggingface segment anything model to mlflow.

        :param task_type: Task type used in training.
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
        if self._task_type == Tasks.MASK_GENERATION.value:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._model = SamModel.from_pretrained(model_dir).to(_map_location)
                self._processor = SamProcessor.from_pretrained(model_dir)
                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data:
            Pandas DataFrame with a first column name ["image"] which is in base64 String format or an URL, and
            second column name ["input_points"] -
            string representation of a numpy array of shape `(point_batch_size, num_points, 2)`:
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 3. The first dimension is the point batch size (i.e.
            how many segmentation masks do we want the model to predict per input point), the third dimension is the
            number of points per segmentation mask (it is possible to pass multiple points for a single mask), and the
            last dimension is the x (vertical) and y (horizontal) coordinates of the point. If a different number of
            points is passed either for each image, or for each mask, the processor will create "PAD" points that
            will correspond to the (0, 0) coordinate, and the computation of the embedding will be skipped for these
            points using the labels.

            third column name ["input_boxes"] -
            string representation of a numpy array of shape `(num_boxes, 4)`:
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the the number of
            boxes per image and the coordinates of the top left and botton right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

            fourth column name ["input_labels"] -
            string representation of a numpy array of shape `(point_batch_size, num_points)`:
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder
            The padding labels should be automatically done by the processor.

            fifth column name ["multimask_output"] -
            boolean value to indicate if the model should return a single mask or multiple masks per input point.
            If False, the model will return a single mask per input point, if True, the model will return multiple
            masks per input point. Default is True.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with generated masks and IOU score corresponding to each inputs
            [input_points, input_boxes, input_labels] converted to base64 string.
        :rtype: pd.DataFrame
        """
        # Do inference one input at a time.
        response = []
        for image, input_points, input_boxes, input_labels, multimask_output in zip(
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_POINTS],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_BOXES],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_LABELS],
            input_data[MLflowSchemaLiterals.INPUT_PARAM_MULTIMASK_OUTPUT],
        ):
            # Decode the image and make a PIL Image object.
            pil_image = Image.open(io.BytesIO(process_image(image)))

            input_points = string_to_nested_float_list(input_points) if input_points else None
            input_boxes = string_to_nested_float_list(input_boxes) if input_boxes else None
            input_labels = string_to_nested_float_list(input_labels) if input_labels else None

            # if input_points, input_boxes and input_labels is not None, make a batch of 1
            # as multiple batches with None input is not supported yet in SAM HF processor
            # todo - remove this when multiple batches with None input is supported in SAM HF processor
            input_points = [input_points] if input_points else None
            input_boxes = [input_boxes] if input_boxes else None
            input_labels = [input_labels] if input_labels else None
            multimask_output = multimask_output if isinstance(multimask_output, bool) else True

            # Do inference.
            _map_location = "cuda" if torch.cuda.is_available() else "cpu"
            processed_inputs = self._processor(
                pil_image,
                input_boxes=input_boxes,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(_map_location)

            with torch.no_grad():
                if multimask_output in [True, False]:
                    outputs = self._model(**processed_inputs, multimask_output=multimask_output)
                else:
                    outputs = self._model(**processed_inputs)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.to(_map_location),
                processed_inputs[SAMHFLiterals.ORIGINAL_SIZES].to(_map_location),
                processed_inputs[SAMHFLiterals.RESHAPE_INPUT_SIZES].to(_map_location),
            )
            scores = outputs.iou_scores

            masks_numpy = masks[0].cpu().numpy()
            scores_list = scores.squeeze(dim=0).tolist()
            masks_numpy_shape = masks_numpy.shape

            pred = {MLflowSchemaLiterals.RESPONSE_DF_PREDICTIONS: []}
            for i in range(masks_numpy_shape[0]):
                per_pred = {MLflowSchemaLiterals.RESPONSE_DF_MASKS_PER_PREDICTION: []}
                for j in range(masks_numpy_shape[1]):
                    per_pred[MLflowSchemaLiterals.RESPONSE_DF_MASKS_PER_PREDICTION].append(
                        {
                            MLflowSchemaLiterals.RESPONSE_DF_ENCODED_BINARY_MASK: image_to_base64(
                                bool_array_to_pil_image(masks_numpy[i][j]), format=DatatypeLiterals.IMAGE_FORMAT
                            ),
                            MLflowSchemaLiterals.RESPONSE_DF_IOU_SCORE: scores_list[i][j],
                        }
                    )
                pred[MLflowSchemaLiterals.RESPONSE_DF_PREDICTIONS].append(per_pred)
            response.append(pred)

        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE: response})
        return df_responses
