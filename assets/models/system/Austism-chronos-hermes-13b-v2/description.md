# **Model Details**
Note: Use of this model is governed by the Meta license. Click on View License above.

The "chronos-hermes-13b-v2" model is a combination of two existing models: "chronos-13b-v2" and "Nous-Hermes-Llama2-13b." The merge ratio is 75% from "chronos-13b-v2" and 25% from "Nous-Hermes-Llama2-13b." This fusion is designed to leverage the imaginative and creative writing style of "chronos-13b-v2" while maintaining coherence and practicality in its outputs. The model is capable of generating long and highly expressive prose, making it suitable for tasks that require creative and coherent text generation. It supports a maximum context length of 4096 tokens, enabling it to handle lengthy and complex inputs for a wide range of applications.



## **Sample inputs and outputs (for real-time inference)**

### **Sample input**
```json
{"input_data": {"input_string": ["My name is John and I am", "Once upon a time,"], "parameters": {"max_new_tokens":100, "do_sample":true}}}
```

### **Sample output**
```json
[
  {
    "0": "My name is John and I am the project leader for the EMPOWER study. This study will explore the feasibility of using a low-energy shock wave to prevent the life-threatening complication of deep vein thrombosis (DVT) in patients undergoing hip or knee replacement surgery.\nThe EMPOWER study is currently recruiting patients who are undergoing hip or knee replacement surgery at the Royal Orthopedic Hospital in Birmingham, UK. The"
  },
  {
    "0": "Once upon a time, the Internet was the new, exciting, revolutionary way of communication that everyone wanted to use.\nWe were giddy with the possibilities. The potential for change and access to knowledge was enormous. But now that we have embraced it, we struggle to control it. What started as the democratization of information and human connection has become a chaotic stream of data, rumors, fear, and resentment. Every new tool and technology that emerges seems to take"
  }
]
```
