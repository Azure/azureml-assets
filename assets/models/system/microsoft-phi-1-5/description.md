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
      "max_new_tokens": 200,
      "do_sample": true
    }
  }
}
```

### Sample output
```json
{
  "output": [
    "What is a fermi paradox? Answer: The fermi paradox is a paradox that arises from the observation that, if there is a high probability of a particle being in a particular state, it should not be observed at all. Exercise 3: What is the Higgs boson? Answer: The Higgs boson is a particle that was discovered in 2012. It gives other particles mass by interacting with them through a fundamental force. Exercise 4: What is the role of particle physics in modern medicine? Answer: Particle physics has contributed to the development of many medical treatments, including radiation therapy for cancer and imaging techniques like MRI and PET scans. Exercise 5: What is the role of particle physics in renewable energy? Answer: Particle physics has contributed to the development of renewable energy technologies like solar cells and wind turbines. Title: The Importance of Comparison: Judicious and Foolhardy Introduction: In our daily lives,"
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
    "Alice: What is a fermi paradox?\nA: The fermi paradox is the paradox that arises from the fact that the probability of the existence of a black hole is incredibly high, yet we cannot observe it directly.\n\nBob: That's fascinating! So, if we can't see the black hole, how can we know it exists?\nAlice: Well, scientists have indirect evidence, such as the gravitational effects on nearby matter and the emission of high-energy radiation.\n\nBob: But if we can't see"
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
    "def is_prime(n: int) -> bool:\n        if n < 2:\n            return False\n        for i in range(2, int(math.sqrt(n)) + 1):\n            if n % i == 0:\n                return False\n        return True\n\n    for i in range(len(li)):\n        for j in range(i + 1, len(li)):\n            if is_prime(li[i] + li[j]"
  ]
}
```
