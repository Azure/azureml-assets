# **Model Details**
The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an instruct fine-tuned version of the Mistral-7B-v0.2.

Mistral-7B-v0.2 has the following changes compared to Mistral-7B-v0.1:
- 32k context window (vs 8k context in v0.1)
- Rope-theta = 1e6
- No Sliding-Window Attention

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/la-plateforme/).

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
        ],
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": true,
            "max_new_tokens": 200,
            "return_full_text": false
        }
    }
}
```

### **Sample output**
```json
[
  {
    "0": " 'Paris is known for its rich history, art, culture, and cuisine. Here are some must-visit places:\n\n1. Eiffel Tower: An iconic landmark and a symbol of Paris, it offers stunning views of the city.\n2. Louvre Museum: Home to thousands of works of art, including the Mona Lisa.\n3. Notre-Dame Cathedral: A stunning Gothic cathedral with beautiful architecture and intricate details.\n4. Montmartre: A hill with a beautiful view of Paris, famous for its artists, cabarets, and the Sacré-Cœur Basilica.\n5. Musée d'Orsay: A museum housing mainly French art from the Impressionist and Post-Impressionist periods.\n6. Seine River: Take a boat ride to enjoy the scenic beauty of Paris from the river.\n7. Champs-Élysées: A",
    "1": "\n\n{'role': 'assistant', 'content': \"I'd be happy to help you plan a trip to Paris! Based on your interests, here are some suggestions for things to do and see:\\n\\nIf you're interested in art, I would recommend visiting the Louvre Museum, Musée d'Orsay, and the Centre Pompidou. These museums offer a wide range of art from different periods and styles.\\n\\nIf you're interested in history, you might want to visit the Palace of Versailles, the Sainte-Chapelle, and the Arc de Triomphe. These sites offer a glimpse into France's rich history and cultural heritage.\\n\\nIf you're looking for a more relaxed experience, you could take a boat tour along the Seine River, or spend an afternoon in one of Paris' many beautiful parks, such as the Luxembourg Gardens or the Jardin des Tuileries",
    "2": "\n\n'I'd be happy to help answer questions about a specific topic or number, but I need more context to understand what you mean by \"#1\". Could you please provide some additional information or clarify your question?'"
  }
]
```
