Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

### Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/phi3.5-techblog) <br>
📖 [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

### Model Summary
|      |      |
|------|------|
| **Architecture** | Phi-3.5-vision has 4.2B parameters and contains image encoder, connector, projector, and Phi-3 Mini language model. |
| **Inputs** | Text and Image. It’s best suited for prompts using the chat format. |
| **Context length** | 128K tokens |
| **GPUs** | 256 A100-80G |
| **Training time** | 6 days |
| **Training data** | 500B tokens (vision tokens + text tokens) |
| **Outputs** | Generated text in response to the input |
| **Dates** | Trained between July and August 2024 |
| **Status** | This is a static model trained on an offline text dataset with cutoff date March 15, 2024. Future versions of the tuned models may be released as we improve models. |
| **Release date** | August 20, 2024 |
| **License** | MIT |
