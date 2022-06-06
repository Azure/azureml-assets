# Built-In Vision Components


## Overview

Using built-in components, you can train a vision model without writing any code. All that's required is labeled input data. You run the component on labeled input data, and you get a model back. If you want, you can also select the model you want to train, model hyperparameters, etc.


## Available Components

![Sample Dataset](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-automated-ml/automl-computer-vision-tasks.png)

Image is from http://cs231n.stanford.edu/slides/2021/lecture_15.pdf.

There are built-in components to train models for the following types of tasks:

1. **Image classification** &ndash; Tasks where an image is classified with one or more labels from a set of classes - e.g. each image can be labeled as 'cat', 'dog', and/or 'duck'
See YAML definition

1. **Object detection** &ndash; Tasks to identify objects in an image and locate each object with a bounding box e.g. locate all dogs and cats in an image and draw a bounding box around each.
See YAML definition

1. **Instance segmentation** &ndash; Tasks to identify objects in an image at the pixel level, drawing a polygon around each object in the image.

You can refer to the YAML definitions of the components below. YAML definitions contain the component name, inputs, and other schema information:
1. [Image classification](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/image_classification/spec.yaml)
1. [Object detection](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/object_detection/spec.yaml)
1. [Instance segmentation](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/instance_segmentation/spec.yaml)


## Formatting Input

Input data (both training and validation data) is formatted as JSONL. Refer to [this document](https://docs.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema) to learn how to format input data for each type. (Note: the article concerns formatting for AutoML, but the input format is the same for AutoML and the built-in components.)


## End-to-End Example

You can see an end-to-end example [here](azureml-examples). The example prepares real input image data, trains an object detection model, and then inferences using the trained model.
