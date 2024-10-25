Phi-3.5-MoE is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available documents - with a focus on very high-quality, reasoning dense data. The model supports multilingual and comes with 128K context length (in tokens). The model underwent a rigorous enhancement process, incorporating supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

### Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/phi3.5-techblog) <br>
📖 [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

### Model Architecture
Phi-3.5-MoE has 16x3.8B parameters with 6.6B active parameters when using 2 experts. The model is a mixture-of-expert decoder-only Transformer model using the tokenizer with vocabulary size of 32,064.

### Training Data
This is a static model trained on an offline dataset with 4.9T tokens and a cutoff date October 2023 for publicly available data. Future versions of the tuned models may be released as we improve models.
