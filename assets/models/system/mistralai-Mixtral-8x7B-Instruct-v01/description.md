The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks with 6x faster inference.

Mixtral-8x7B-v0.1 is a decoder-only model with 8 distinct groups or the "experts". At every layer, for every token, a router network chooses two of these experts to process the token and combine their output additively. Mixtral has 46.7B total parameters but only uses 12.9B parameters per token using this technique. This enables the model to perform with same speed and cost as 12.9B model.

For full details of this model please read [release blog post](https://mistral.ai/news/mixtral-of-experts/).

# Limitations and Biases

The Mixtral-8x7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance.

It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

# Sample inputs and outputs

### Sample input
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
        ],
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": true,
            "max_new_tokens": 200,
            "return_full_text": true
        }
    }
}
```

### Sample output
```json
{
  "output": "The Eiffel Tower is one of the most iconic landmarks in the world and is a must-see attraction in Paris. Here are a few reasons why the Eiffel Tower is so great:\n\n1. Iconic symbol: The Eiffel Tower is instantly recognizable and is often used as a symbol of Paris and France.\n2. Breathtaking views: The Eiffel Tower offers stunning views of the city, and there are several different viewing platforms to choose from, including the top floor, which is located 906 feet (276 meters) above the ground.\n3. Romantic atmosphere: The Eiffel Tower is often associated with romance and is a popular spot for proposals and weddings.\n4. Historical significance: The Eiffel Tower was built for the 1889 World's Fair and was originally intended to be a temporary structure. However, it has since become a permanent fixture in the Paris"
}
```
