# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--intel-extension", action="store_true")
args = parser.parse_args()

# Intel(R) Extension for PyTorch* cpu package is optimized for Intel CPU
device = "cpu"

# Create dummy dataset
texts = [
    "This is a positive sentence.",
    "This is a negative sentence.",
    "I'm feeling great today.",
    "I'm not happy with the outcome.",
]
labels = [1, 0, 1, 0]

# Tokenizer data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Create dataloader
dataset = torch.utils.data.TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"], torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-3)

# Optimize using Intel(R) Extension for PyTorch*
if (args.intel_extension):
    print("Intel Optimizations Enabled")
    import intel_extension_for_pytorch as ipex
    model, optimizer = ipex.optimize(model,optimizer=optimizer)

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0

    for batch in dataloader:
        input_ids, attention_mask, target = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")