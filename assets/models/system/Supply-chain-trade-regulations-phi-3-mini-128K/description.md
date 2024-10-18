# Description

The Supply-chain-trade-regulations-phi-3 mini-128K is a 3.8B parameters, lightweight, state-of-the-art open model trained using synthetic Supply Chain domain specific datasets that focuses on the sub-domain of Trade regulations. The model is fine-tuned on the Phi-3-Mini-128K-instruct as the base model. The training dataset includes both synthetic data and filtered publicly available data, with an emphasis on high-quality and reasoning-dense properties.

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures. When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, this fine-tuned model did not degrade compared to its baseline performance.

# Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3)

📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)

📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report)

🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai)

👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)

# Model Architecture

Phi-3 Mini-128K-Instruct-Supply-Chain has 3.8B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.


# Training Datasets

The training dataset includes a wide range of books and materials collected from the web:

	1. Publicly available books in general supply chain domains that are processed to create Q&A pairs using GPT-4o
	2. Publicly available books in the Trade Compliance domain, including codes and regulations (Federal and International)
	3. Verified question-answer sets commonly asked by trade compliance managers on trade regulation, scaled synthetically
	3. Publicly available FineWeb-Edu dataset filtered to supply chain domain documents

# License

The model is licensed under the MIT license.


# Intended Uses

## Primary use cases
The model is intended to be used as for scenarios related to import/export operations, trade sanctions and export control and tariffs and duty management, that require analyzing trade codes and regulations.Additionally, the model trained on  wide range of professional materials in the supply chain domain, can be used to query supply chain related textual data, for example in the scenario of training and guiding personnel in the supply chain domain.

We recommend using the model in combination with RAG pipeline to ground the responses on up to date, relevant information.

## Out-of-scope use cases

The model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios.

The model is not designed to provide any professional advice or recommendations.

Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

# Responsible AI Considerations

Like other language models, the Phi series models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:

	• Quality of Service: the Phi models are trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.

	• Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.

	• Inappropriate or Offensive Content: these models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.

	• Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

	• Limited Scope for Code: Majority of Phi-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:
	• Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.

	• High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.

	• Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).

	• Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.

	• Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

# Benchmarks
	- Supply Chain Categories
	- Trade Compliance Categories
	- GSM8K
	- AgiEval-en
	- MMLU

# Hardware

Note that by default, the Phi-3 Mini-4K-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
	• NVIDIA A100
	• NVIDIA A6000
	• NVIDIA H100
If you want to run the model on:
	• NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"
	• CPU: use the GGUF quantized models 4K
	• Optimized inference on GPU, CPU, and Mobile: use the ONNX models 4K
