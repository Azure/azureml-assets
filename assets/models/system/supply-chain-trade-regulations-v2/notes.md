<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `note.md` is highly recommended, but not required. It captures information about how your model is created. We highly recommend including this section to provide transparency for the customers. -->

## Intended Use

### Primary Use Cases

The model is intended to be used as for scenarios related to Import/Export operations, Trade Sanctions, Export Control and Tariffs and Duty Management, that require analyzing trade codes and regulations. Additionally, the model is trained on a wide range of professional materials in the supply chain domain, and can be used to query supply chain related textual data, for example in the scenario of training and guiding personnel in the supply chain domain.

We recommend using the model in combination with a Retrieval Augmented Generation (RAG) pipeline to ground the responses on up to date, relevant information. For best results in querying tables, input should be clean, preferably in a markdown format.

### Out-of-Scope Use Cases

The model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios.

The model is not designed for Supply Chain optimization, forecasting or database query generation.

The model is not designed to provide any professional advice or recommendations.

Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

**Nothing written here should be interpreted as or deemed a restriction or modification to the license the model is released under.**

## Responsible AI Considerations

The adapted-AI-model-for-supply-chain-trade-regulations-analysis model, like other language models, can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:

- Quality of Service: The Phi models are trained primarily on English text and some additional multilingual text. Languages other than English will experience worse performance as well as performance disparities across non-English. English language varieties with less representation in the training data might experience worse performance than standard American English.
- Multilingual performance and safety gaps: We believe it is important to make language models more widely available across different languages, but the Phi 3 models still exhibit challenges common across multilingual releases. As with any deployment of LLMs, developers will be better positioned to test for performance or safety gaps for their linguistic and cultural context and customize the model with additional fine-tuning and appropriate safeguards.
- Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups, cultural contexts, or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.
- Inappropriate or Offensive Content: These models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
- Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

Developers should apply responsible AI best practices, including mapping, measuring, and mitigating risks associated with their specific use case and cultural, linguistic context. Phi-3 family of models are general purpose models. As developers plan to deploy these models for specific use cases, they are encouraged to fine-tune the models for their use case and leverage the models as part of broader AI systems with language-specific safeguards in place. Important areas for consideration include:

- Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
- High-Risk Scenarios: Developers should assess the suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
- Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
- Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
- Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

## Training Data

The adapted-AI-model-for-supply-chain-trade-regulations-analysis model was trained on materials in the Supply Chain Trade Compliance domain collected from the web, totaling tens of millions of tokens. The training data includes:

	1. Publicly available documents in the Trade Compliance domain, including tariff codes, duty rates, regulations (with global/regional/local coverage) and authority rulings
	2. Verified question-answer sets commonly asked by trade compliance managers on trade regulation, scaled synthetically

## Hardware

Note that by default, the Phi\-4 model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA H100

## Disclaimer

Customer agrees that any information or output resulting from the use of the adapted-AI-model-for-supply-chain-trade-regulations-analysis is for informational or internal process management purposes only, and does not constitute legal, trade compliance, or other professional advice from Microsoft. Customer agrees that it is responsible for its own trade compliance research and decisions, and that the solutions and resulting information provided through the Adapted-AI-model-for-supply-chain-trade-regulations-analysis will not serve as the primary basis for its trade compliance decisions. Customer agrees that Microsoft is not responsible or liable for any decisions or actions Customer, or its authorized third parties, take based on information Customer produces or generates as a user of the Adapted-AI-model-for-supply-chain-trade-regulations-analysis. No solutions provided through the Adapted-AI-model-for-supply-chain-trade-regulations-analysis constitute an offer, solicitation of an offer, or advice to engage in or refrain from any trade-related activities, or any other trade compliance actions by Microsoft.

## License

The model is licensed under the MIT license.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Resources

[https://ai.azure.com/explore/models/Phi\-4/version/7/registry/azureml?tid\=72f988bf\-86f1\-41af\-91ab\-2d7cd011db47](https://ai.azure.com/explore/models/Phi-4/version/7/registry/azureml?tid=72f988bf-86f1-41af-91ab-2d7cd011db47)
