# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import mlflow.tensorflow

mlflow.tensorflow.autolog()

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the open-source dataset from the "datasets" library
dataset = load_dataset('glue', 'mrpc')

# Prepare the dataset
train_data = dataset['train']
eval_data = dataset['validation']

# Extract the text and label fields from the dataset
train_texts = train_data['sentence1']
train_labels = train_data['label']

eval_texts = eval_data['sentence1']
eval_labels = eval_data['label']

# Tokenize and encode the training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

# Create TensorFlow datasets for training and evaluation
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).batch(16)

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(eval_encodings),
    eval_labels
)).batch(16)

# Load the pre-trained BERT model
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=3, batch_size=16)

# Evaluate the model
model.evaluate(eval_dataset, verbose=2)