[Model catalog \- Azure AI Studio](https://ai.azure.com/explore/models/Phi-3-small-128k-instruct/version/3/registry/azureml?tid=72f988bf-86f1-41af-91ab-2d7cd011db47)

**Fin\-report\-analysis\-phi\-3\-small\-128K**

### Description

Fin\-report\-analysis\-phi\-3\-small\-128K is a state\-of\-the\-art small language model (SLM) based on the Phi\-3\-small\-128k architecture, designed specifically for analyzing financial reports. It has been fine\-tuned on a few hundred million tokens derived from instruction data over financial documents, including SEC filings (10\-K, 10\-Q, 8\-K reports) and mathematical reasoning tasks. The model is optimized to handle complex financial language and table understanding, making it suitable for SEC reports analysis, including data extraction, summarization, and common financial formulas. It is also capable of more complex reasoning tasks, such as comparing companies and identifying trends across different time periods. 

### Model Architecture

Fin\-report\-analysis\-phi\-3\-small\-128K is a dense, decoder\-only Transformer model with 7B parameters, optimized for the task of financial reports analysis. It supports a 128K context length, making it capable of processing long financial documents and providing coherent, context\-aware completions. The model is fine\-tuned with Supervised fine\-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

### Training Datasets

 Fin\-report\-analysis\-phi\-3\-small\-128K was fine\-tuned on a highly specialized dataset totaling few hundred million tokens, which includes:

* SEC filings, including 10\-K, 10\-Q, and 8\-K reports
* Textbook\-like materials focusing on finance
* Verified common questions and answer pairs on SEC reports, scaled synthetically
* Mathematical reasoning tasks

The training data was carefully selected to ensure that the model excels in financial reasoning, risk assessment, and understanding complex financial terms and scenarios.

### License

The model is licensed under the MIT license.

### Intended Uses

Primary Use Cases:

The model is intended for scenarios that require financial report analysis in English, focusing on 10\-K, 10\-Q, 8\-K and equivalents . It is particularly suited for general financial AI systems and applications that require:

* Financial tables understanding
* Extraction and summarization of information from financial documents
* Answering questions related to SEC reports, such as risk assessment and analysis of companies’ financial performance.

We recommend using the model in combination with RAG pipeline to ground the responses on up to date, relevant information. The model was trained on chunks from SEC reports.

* Extraction of key financial entities

Out\-of\-Scope Use Cases:

The model is not specifically designed or evaluated for all downstream purposes. The model is not designed to provide any financial advice or recommendations. Developers should be mindful of common limitations of language models when selecting use cases and ensure that they evaluate and mitigate potential issues related to accuracy, safety, and fairness before deploying the model, particularly in high\-risk scenarios. Additionally, developers should adhere to relevant laws and regulations (including privacy and trade compliance laws) applicable to their use cases.

**Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.**

### Responsible AI Considerations

 Fin\-report\-analysis\-phi\-3\-small\-128K like other language models, can potentially exhibit biases or generate outputs that may not be suitable for all contexts. Developers should be aware of the following considerations:

* Quality of Service: the Fin\-report\-analysis\-phi\-3\-small\-128K model is trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.
* Representation of Harms \& Perpetuation of Stereotypes: This model can over\- or under\-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post\-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real\-world patterns and societal biases.
* Inappropriate or Offensive Content: this model may produces other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
* Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:

* Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
* High\-Risk Scenarios: Developers should assess suitability of using models in high\-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
* Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end\-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use\-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
* Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
* Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

### Content Filtering

Prompts and completions are passed through a default configuration of Azure AI Content Safety classification models to detect and prevent the output of harmful content. Learn more about [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview). Configuration options for content filtering vary when you deploy a model for production in Azure AI; [learn more](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview).

Benchmarks

**Finetuning Samples**

 

**Sample Inputs \& Outputs (for real\-time inference)**

 

 

 

**Hardware**

 

Note that by default, the Phi\-3\-small\-128K\-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100

If you want to run the model on:

* NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from\_pretrained() with attn\_implementation\="eager"
* CPU: use the GGUF quantized models [4K](https://aka.ms/Phi3-mini-4k-instruct-gguf)
* Optimized inference on GPU, CPU, and Mobile: use the ONNX models [4K](https://aka.ms/Phi3-mini-4k-instruct-onnx)

 

