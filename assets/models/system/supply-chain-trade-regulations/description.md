**Description**

The adapted AI model for supply chain trade regulations analysis (preview) is a 3\.8B parameter, lightweight, state\-of\-the\-art open model, trained using synthetic supply chain domain\-specific datasets, focused on trade regulations. 

The model is fine\-tuned on the base model, Phi\-3\-Mini\-128K\-instruct. The training dataset includes both synthetic data and filtered publicly available data, with an emphasis on high\-quality and reasoning\-dense properties.

The model underwent a post\-training process that incorporates both supervised fine\-tuning and direct preference optimization for following instructions and safety measures. This fine\-tuned model did not degrade compared to its baseline performance for common sense, language understanding, math, code, long context, or logical reasoning when assessed against benchmarks.

*NOTE: This model is in preview*

**Disclaimer**

One should not make any decisions based on the output of the model. Please review the output content to make sure it's accurate before relying on such output.

**Resources**

Here are some links to resources related to the Phi\-3 class of models:

🏡 [Phi\-3 Portal](https://azure.microsoft.com/en-us/products/phi-3)

📰 [Phi\-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)

📖 [Phi\-3 Technical Report](https://aka.ms/phi3-tech-report)

🛠️ [Phi\-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai)

👩‍🍳 [Phi\-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)

**Model Architecture**

The adapted AI model for supply chain trade regulations analysis has 3\.8B parameters and is a dense decoder\-only transformer model. The model is fine\-tuned with supervised fine\-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

**Training Datasets**

The training dataset includes a wide range of books and materials collected from the web:

1. Publicly available books in general supply chain domains that are processed to create Q\&A pairs using GPT\-4o
2. Publicly available books in the trade compliance domain, including codes and regulations (federal and international)
3. Verified question\-answer sets commonly asked by trade compliance managers on trade regulation, scaled synthetically
4. Publicly available FineWeb\-Edu dataset filtered to supply chain domain documents

**License**

The model is licensed under the MIT license.

**Intended Uses**

**Primary use cases**

The model is intended to be used for scenarios related to import/export operations, trade sanctions and export control, and tariffs and duty management that require analyzing trade codes and regulations. The model was also trained on a wide range of professional materials in the supply chain domain and can be used to query supply chain related textual data, for example training and guiding supply\-chain personnel.

We recommend using the model in combination with RAG pipeline to ground the responses on up\-to\-date, relevant information. For best results querying tables, input should be clean, preferably in a markdown format.

**Out\-of\-scope use cases**

The model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high\-risk scenarios.

The model is not designed for supply chain optimization, forecasting, or database query generation.

The model is not designed to provide any professional advice or recommendations.

Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

Nothing written here should be interpreted as or deemed a restriction or modification to the license the model is released under.

**Trademarks**

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark \& Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third\-party trademarks or logos are subject to those third\-party's policies.

**Responsible AI Considerations**

Like other language models, the Phi series models can behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:

• Quality of Service: The Phi models are trained primarily on English text. Languages other than English will not perform as well. English language varieties with less representation in the training data might not perform as well as standard American English.

• Representation of Harms \& Perpetuation of Stereotypes: These models can over\- or under\-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post\-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real\-world patterns and societal biases.

• Inappropriate or Offensive Content: These models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.

• Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

• Limited Scope for Code: Majority of Phi\-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (for example. privacy, trade, etc.). Important areas for consideration include:

• Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (for example: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.

• High\-Risk Scenarios: Developers should assess suitability of using models in high\-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (for example: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.

• Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end\-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use\-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).

• Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.

• Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

**Benchmarks**

We provide two types of benchmark results, the first one is specific to the Trade Compliance domain and based on in\-house evaluation dataset created by subject matter experts. The second one is based on general publicly available datasets that evaluate language understanding, math skill, etc.

**Description**

The Trade Compliance evaluation dataset consists of four sections:

1. Tariff \& Duty section contains questions related to the product code and import duty rate specific to a country/region, (e.g., European Union, USA, etc.), e.g., What is the EU TARIC for a smartphone?
2. Import\-Export: examples based on the rules and regulations related to import/export, e.g., What is the purpose of the US CTPAT Trade Compliance Program?
3. Policy \& Training: examples related to Trade Compliance, e.g., Which countries/regions do we consider as EU customs territory?
4. Trade Sanctions: examples related to current market condition, e.g., How can I use the Commerce Country Chart when I have the ECCN?

**Performance**



| **Category** | **GPT\-4o** | **GPT\-4o\-mini** | **Phi\-3\-mini\-128K\-Ins** | **Adapted\-AI\-model\-for\-supply\-chain\-trade\-regulations\-analysis** |
| --- | --- | --- | --- | --- |
| Tariffs \& Duty | 1\.77 | 1\.67 | 1\.24 | 1\.65 |
| Import\-Export | 1\.49 | 1\.45 | 1\.34 | 1\.48 |
| Policy \& Training | 1\.82 | 1\.75 | 1\.63 | 1\.76 |
| Trade Sanction | 1\.49 | 1\.35 | 1\.02 | 1\.35 |
| Overall (normalized to 1\) | 0\.82 | 0\.78 | 0\.65 | 0\.78 |

**General benchmarking**



| **Category** | **Benchmark** | **Phi\-3\-mini\-128K\-Ins** | **Adapted\-AI\-model\-for\-supply\-chain\-trade\-regulations\-analysis** |
| --- | --- | --- | --- |
| Popular aggregated benchmark | AGI Eval | 39\.5 | 37\.8 |
|  | MMLU | 69\.7 | 69\.0 |
| Language Understanding | ANLI | 52\.3 | 52\.8 |
| Reasoning | PIQA | 80\.1 | 78\.6 |
|  | WinoGrande | 71\.0 | 72\.0 |
| Math | GSM\-8K | 85\.3 | 76\.3 |

**Hardware**

Note that by default, the Phi\-3 Mini\-4K\-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100