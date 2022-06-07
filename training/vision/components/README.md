# Built-In Vision Components


## Overview

Using built-in components, you can train a computer vision model without writing any code. All that's required to train a model is labeled input data.

Components expose knobs to tune their behavior. If desired, you can specify which model a component trains (like YOLOv5, Faster R-CNN with a ResNet-50-FPN backbone, etc.), model hyperparameters (like learning rate and batch size), and other settings.


## Available Components

There are built-in components to train models for the following types of tasks:

1. **Image classification** &ndash; Images are classified with one or more labels from a set of classes - e.g. each image can be labeled as 'cat', 'dog', and/or 'duck'

1. **Object detection** &ndash; Identify objects in images and locate each object with a bounding box - e.g. locate all dogs and cats in an image and draw a bounding box around each.

1. **Instance segmentation** &ndash; Identify objects in images at the pixel level, drawing a polygon around each object.

You can refer to the YAML definitions of the components below. YAML definitions contain the component name, inputs, and other schema information:
1. [Image classification](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/image_classification/spec.yaml)
1. [Object detection](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/object_detection/spec.yaml)
1. [Instance segmentation](https://github.com/Azure/azureml-assets/blob/main/training/vision/components/instance_segmentation/spec.yaml)


## Formatting Input

Input datasets (both training and validation datasets) are formatted as JSONL files. Refer to [this document](https://docs.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema) to learn how to format input data for each type. (Note: the article concerns formatting for AutoML, but the input format is the same for AutoML and the built-in components.)


## End-to-End Example

You can see an end-to-end example [here](https://github.com/Azure/azureml-examples/tree/vision-components-preview/cli/jobs/training/vision/object-detection-using-built-in-component). The example prepares real input image data, trains an object detection model, and then inferences using the trained model.
