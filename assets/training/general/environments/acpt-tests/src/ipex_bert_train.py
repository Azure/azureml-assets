# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.integrations import MLflowCallback
from datasets import load_dataset
import argparse
import mlflow
import time

parser = argparse.ArgumentParser()
parser.add_argument("--intel-extension", action="store_true")
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset = load_dataset('glue', 'mrpc')
train_dataset, test_dataset = dataset['train'], dataset['validation']

# Load the tokenizer and encode the data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        max_length=128,  # Set the maximum sequence length
        padding='max_length'  # Pad all sequences to the same length
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
model.train()

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.8, 0.999), weight_decay=3e-07)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

if (args.intel_extension):
    # Optimize using Intel(R) Extension for PyTorch*
    print("Intel Optimizations Enabled")
    import intel_extension_for_pytorch as ipex
    model, optimizer = ipex.optimize(model,optimizer=optimizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='outputs',
    num_train_epochs=5,
    per_device_train_batch_size=93,
    per_device_eval_batch_size=93,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=1000,
    evaluation_strategy='epoch',
    eval_steps=100,
    learning_rate=3e-05
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=[optimizer,scheduler]
)

trainer.pop_callback(MLflowCallback)
start = time.time()

# Train the model
result = trainer.train()

print(f"Time: {result.metrics['train_runtime']:.2f}")
print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
print("Training...")

mlflow.log_metric(
    "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
)

# Evaluate the model 
print("Evaluation...")
trainer.evaluate()
