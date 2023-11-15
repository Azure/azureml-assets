# **Model Details**

The Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) generative text model using a variety of publicly available conversation datasets.

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

This instruction model is based on Mistral-7B-v0.1, a transformer model with the following architecture choices:
- Grouped-Query Attention
- Sliding-Window Attention
- Byte-fallback BPE tokenizer

## Limitations

The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. 
It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

# **Inference samples**

## **Sample inputs and outputs (for real-time inference)**

### **Sample input**
```json
{
  "input_data": {
    "input_string": [
      {
        "role": "user",
        "content": "I am going to Paris, what should I see?"
      },
      {
        "role": "assistant",
        "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."
      },
      {
        "role": "user",
        "content": "What is so great about #1?"
      }
    ]
  },
  "params": {
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": true,
    "max_new_tokens": 200
  }
}
```

### **Sample output**
```json
{
  "output": "The Eiffel Tower is one of the most iconic landmarks in the world and is considered a symbol of Paris and France. It was built in 1889 for the Exposition Universelle (World's Fair) to celebrate the 100th anniversary of the French Revolution and has since become a popular tourist attraction.\n\nOne of the things that makes the Eiffel Tower so great is its unique design. It was designed by Gustave Eiffel and his company and is made of wrought iron. The tower stands at 324 meters tall, including its antenna, and is made up of three levels that visitors can access.\n\nAnother thing that makes the Eiffel Tower great is the views it offers. From the top, visitors can see the entire city of Paris, as well as the surrounding countryside. The tower also offers a unique perspective on the city's architecture and is a great way to get a sense of the city's layout.\n\nOverall, the Eiffel Tower is a beloved landmark in Paris and is considered a must-see attraction for anyone visiting the city."
}
```
