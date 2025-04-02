<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

The **adapted AI model for supply chain trade regulations analysis** (preview) is a 14B parameters, lightweight, state-of-the-art open model trained using synthetic Supply Chain domain specific datasets that focuses on the sub-domain of Trade regulations. The model is fine-tuned on Phi-4 as the base model. The training dataset comprises synthetically generated data, with an emphasis on high-quality and complex questions with reasoning-dense properties.

The adapted AI model for supply chain trade regulations analysis is a dense decoder-only Transformer model which underwent continual pre-training on Trade Compliance documents for domain knowledge injection, followed by supervised fine-tuning for instruction following. The fine-tuned model is capable of answering questions on relevant context from the Supply Chain domain, including information extraction, summarization and complex reasoning tasks (involving multiple sub-queries with tariff code and duty rate identification and associated numerical calculations). When assessed against general benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, this fine-tuned model did not degrade compared to the baseline model performance. For reference, the Phi-4 Technical Report can be found [here](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/P4TechReport.pdf).

*NOTE: This model is in preview*
