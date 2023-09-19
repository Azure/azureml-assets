# **Microsoft Phi-1.5**

Phi-1.5 is a Transformer-based language model with 1.3 billion parameters. It was trained on a combination of data sources, including an additional source of NLP synthetic texts. Phi-1.5 performs exceptionally well on benchmarks testing common sense, language understanding, and logical reasoning among models with less than 10 billion parameters. The model is open-source and intended for research purposes to explore safety challenges in language models.

## Intended Uses

Phi-1.5 is best suited for prompts using the QA format, the chat format, and the code format. 
Note: that phi-1.5, being a base model, often produces irrelevant text following the main answer

## Limitations

* Generate Inaccurate Code and Facts: The model often produces incorrect code snippets and statements. Users should treat these outputs as suggestions or starting points, not as definitive or accurate solutions.
* Limited Scope for code: If the model generates Python scripts that utilize uncommon packages or scripts in other languages, we strongly recommend users manually verify all API uses.
* Unreliable Responses to Instruction: The model has not undergone instruction fine-tuning. As a result, it may struggle or fail to adhere to intricate or nuanced instructions provided by users.
* Language Limitations: The model is primarily designed to understand standard English. Informal English, slang, or any other language outside of English might pose challenges to its comprehension, leading to potential misinterpretations or errors in response.
* Potential Societal Biases: Regardless of the safe data used for its training, the model is not entirely free from societal biases. There's a possibility it may generate content that mirrors these societal biases, particularly if prompted or instructed to do so. We urge users to be aware of this and to exercise caution and critical thinking when interpreting model outputs.
* Toxicity: Despite that the model is trained with carefully selected data, the model can still produce harmful content if explicitly prompted or instructed to do so. We chose to release the model for research purposes only -- We hope to help the open-source community develop the most effective ways to reduce the toxicity of a model directly after pretraining.

**Training:**

* The model was trained with 30 billion tokens, including 150 billion training tokens, using 32 GPUs over 8 days.
* Software used includes PyTorch, DeepSpeed, and flash-attention.

**License:**

The model is licensed under the <a href="https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx" target="_blank">Research License</a>.

# Inference samples

## Sample inputs and outputs (for real-time inference)

### Sample Question-Answering input
```json
{
  "input_data": {
    "input_string": [
      "What is a fermi paradox?"
    ],
    "parameters": {
      "top_p": 0.9,
      "temperature": 0.6,
      "max_new_tokens": 100,
      "do_sample": true
    }
  }
}
```

### Sample output
```json
{
  "output": [
    "What is a fermi paradox?\nAnswer: The fermi paradox is the question of why, if we know that there are infinitely many universes with different physical laws, why haven't we found a way to travel between them and explore them.\n\nExercise 2:\nWhat is a black hole?\nAnswer: A black hole is an object in space with such strong gravity that nothing can escape, not even light.\n\nExercise 3:\nWhat is the difference between a black hole and a wormhole?"
  ]
}
```

### Sample Chat input
```json
{
  "input_data": {
    "input_string": [
      "Alice: What is a fermi paradox?"
    ],
    "parameters": {
      "top_p": 0.9,
      "temperature": 0.6,
      "max_new_tokens": 100,
      "do_sample": true
    }
  }
}
```

### Sample output
```json
{
  "output": [
    "Alice: What is a fermi paradox?\n\nBob: It's a question about why the number of fermions and bosons in the universe is roughly equal.\n\nAlice: Oh, I see. So, why are there more fermions than bosons?\n\nBob: Well, there are a few reasons. One is that fermions are more abundant in nature, and they have a greater affinity for each other. Another reason is that fermions can form more stable structures than bosons, pregnancies,"
  ]
}
```


### Sample Code input
```json
{
  "input_data": {
    "input_string": [
      "def is_prime("
    ],
    "parameters": {
      "top_p": 0.9,
      "temperature": 0.6,
      "max_new_tokens": 100,
      "do_sample": true
    }
  }
}
```

### Sample output
```json
{
  "output": [
    "def is_prime(n: int) -> bool:\n        if n < 2:\n            return False\n        for i in range(2, int(math.sqrt(n)) + 1):\n            if n % i == 0:\n                return False\n        return True\n\n    def is_triangular(n: int) -> bool:\n        return is_prime(n * (n + 1) // 2)\n\n    triangular_numbers = [i * ("
  ]
}
```
