**Description**

The adapted AI model for financial reports analysis (preview) is a state\-of\-the\-art small language model (SLM) based on the Phi\-3\-small\-128k architecture, designed specifically for analyzing financial reports. It has been fine\-tuned on a few hundred million tokens derived from instruction data over financial documents, including SEC filings (10\-K, 10\-Q, 8\-K reports) and mathematical reasoning tasks. 

The model is optimized to handle complex financial language and to understand data contained in tables, making it suitable for SEC report analysis, including data extraction, summarization, and common financial formulas. It can also perform more complex reasoning tasks, such as comparing companies and identifying trends across different time periods.

*NOTE: This model is in preview*

**Model Architecture**

The adapted AI model for financial reports analysis is a dense, decoder\-only transformer model with 7B parameters, optimized for financial reports analysis. It supports a 128K context length, making it capable of processing long financial documents and providing coherent, context\-aware completions. The model is fine\-tuned with supervised fine\-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

**Training Datasets**

The adapted AI model for financial reports analysis was fine\-tuned on a highly specialized dataset totaling a few hundred million tokens, including:

* SEC filings, including 10\-K, 10\-Q, and 8\-K reports
* Textbook\-like materials focusing on finance
* Verified common questions and answer pairs on SEC reports, scaled synthetically
* Mathematical reasoning tasks

The training data was carefully selected to ensure that the model excels at financial reasoning, risk assessment, and understanding complex financial terms and scenarios.

**License**

The model is licensed under the MIT license.

**Intended Uses**

Primary Use Cases:

The model is intended for scenarios that require financial report analysis in English, focusing on 10\-K, 10\-Q, 8\-K and equivalents. It’s particularly well\-suited for general financial AI systems and applications that require:

* Financial table understanding
* Extraction and summarization of information from financial documents
* Answering questions related to SEC reports, such as risk assessment and analysis of companies’ financial performance

We recommend using the model in combination with RAG pipeline to ground the responses on up\-to\-date, relevant information. The model was trained on chunks from SEC reports. For best results querying SEC reports, use the following formatting techniques.

Recommended preprocessing for SEC reports

Preprocess of data to be used in our corpus \-

1. Getting the data
2. Splitting the data (chunking)
	1. Chunk
	2. Saving metadata
3. Processing the text
4. Adding headers

Getting the Data:

We recommend using HTML format (available from SEC EDGAR website)

Chunking the Data:

1. Split your HTML filing into chunks– recommended chunk by page
	1. Save the page number as metadata
2. Occasionally pages may contain several sections (mostly referred as Items in SEC filing).
	1. We recommend further chunking those by section
	2. Save section name as metadata

Processing the text:

We recommend handling tabular data and free text differently

1. Convert any free text (excluding tables see 4\.) to markdown using any of the markdown tools available (edgartools [dgunning/edgartools: Navigate SEC Edgar data in Python](https://github.com/dgunning/edgartools), Markdownify, or any other available method).
2. Keep all tables in HTML format. Strip all styling attributes except colspan/rowspan attributes, as they are needed to understand if a table header covers several columns or rows.

Adding headers:

1. Due to the nature of the questions that refer to chunks from different documents across various companies and periods of time, we found that adding a header with a brief title based on the metadata of the chunk as described above (company name, reference period and type of document) into the content of the chunk as part of the prompt improves model performance.

Out\-of\-Scope Use Cases:

The model is not specifically designed or evaluated for all downstream purposes. The model is not designed to provide any financial advice or recommendations. Developers should be mindful of common limitations of language models when selecting use cases and ensure that they evaluate and mitigate potential issues related to accuracy, safety, and fairness before deploying the model, particularly in high\-risk scenarios. Additionally, developers should adhere to relevant laws and regulations (including privacy and trade compliance laws) applicable to their use cases.

**Nothing written here should be interpreted as or deemed a restriction or modification to the license the model is released under.**

**Responsible AI Considerations**

Adapted\-AI\-model\-for\-financial\-reports\-analysis\-preview, like other language models, can exhibit biases or generate outputs that may not be suitable for all contexts. Developers should be aware of the following considerations:

* Quality of Service: The adapted AI model for financial reports analysis model is trained primarily on English text. Languages other than English do not perform as well. English language varieties with less representation in the training data might not perform as well as standard American English.
* Representation of Harms \& Perpetuation of Stereotypes: This model can over\- or under\-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post\-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real\-world patterns and societal biases.
* Inappropriate or Offensive Content: This model may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
* Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (for example. privacy, trade, etc.). Important areas for consideration include:

* Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (for example: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
* High\-Risk Scenarios: Developers should assess the suitability of using models in high\-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (for example: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
* Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end\-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use\-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
* Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
* Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

**Content Filtering**

Prompts and completions are passed through a default configuration of Azure AI Content Safety classification models to detect and prevent the output of harmful content. Learn more about [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview). Configuration options for content filtering vary when you deploy a model for production in Azure AI; [learn more](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview).

**Benchmarks**

The SECQUE Benchmark was developed to evaluate fine\-tuned SLMs in the context of real\-world financial industry applications, specifically targeting their efficacy in assisting financial analysts. It focuses on core scenarios in which SLMs could provide significant value to financial analysts, who analyze information from SEC filings, mainly 10\-K and 10\-Q reports.

The benchmark consists of open\-ended questions designed to reflect real\-world queries posed by financial analysts on SEC filings. Each question entry includes a complex, jargon\-laden query, supporting data, ground\-truth answer, and references to specific sections within the filings. The dataset covers multiple documents and companies, with each question verified to ensure objectivity and dependence solely on the reference data provided, without external information. The questions are categorized into four key task types aligned with common financial analysis activities: risk analysis, ratio analysis, comparative analysis, and insight generation.



| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis** | **Phi3\-small\-128k** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| SECQUE | 73\.8 | 58\.2 | 66\.8 | 78\.1 |

Financial benchmarks (classification):



| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis** | **Phi3\-small\-128k** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| Twitter SA | 85\.6 | 70 | 73\.9 | 80\.4 |
| Twitter Topics | 87 | 48\.6 | 61\.7 | 63\.8 |
| FiQA SA | 75\.4 | 80\.8 | 77\.4 | 78\.2 |
| FPB | 79\.6 | 72\.7 | 78\.4 | 82\.8 |
| **Average F1** | 81\.9 | 68 | 72\.8 | 76\.3 |

Financial benchmarks (exact match)



| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis** | **Phi3\-small\-128k** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| ConvFinQA | 76\.2 | 71\.1 | 78\.3 | 75\.4 |
| FinQA | 66\.1 | 63\.5 | 68\.9 | 69\.9 |
| TACT | 64\.5 | 58\.9 | 66\.1 | 71 |
| **Average exact match** | 68\.9 | 64\.5 | 71\.1 | 72\.1 |

General knowledge benchmarks (comparison with base model):



| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis** | **Phi3\-small\-128k** | **% Difference** |
| --- | --- | --- | --- |
| TriviaQA | 76\.2 | 71\.1 | 0% |
| MedQA | 66\.1 | 63\.5 | 3\.5% |
| MMLU | 64\.5 | 58\.9 | \-1\.3% |
| PIQA | 68\.9 | 64\.5 | 1\.1% |
| WinoGrande | 79 | 80 | \-1\.2% |

\*All evaluations were conducted using temperature 0\.3

 

**Hardware**

 

Note that by default, the Phi\-3\-small\-128K\-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100

**Disclaimer**

Customer agrees that any information or output resulting from the use of the Adapted\-AI\-model\-for\-financial\-reports\-analysis is for informational or internal process management purposes only, and does not constitute legal, financial, tax planning, or other advice from Microsoft. Customer agrees that it is responsible for its own financial research and financial decisions, and that the solutions and resulting information provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis will not serve as the primary basis for its financial decisions. Customer agrees that Microsoft is not responsible or liable for any decisions or actions customer, or its authorized third parties, take based on information Customer produces or generates as a user of the Adapted\-AI\-model\-for\-financial\-reports\-analysis. No solutions provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis constitute an offer, solicitation of an offer, or advice to buy or sell securities, or any financial instrument or investment by Microsoft.

Customer may not use any of the features or information provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis as a factor in establishing the financial standing, including the eligibility for credit, hire, insurance, housing, employment or other eligibility or entitlement (including for any other use constituting a permissible purpose under the U.S. Federal Fair Credit Reporting Act (“FCRA”)) of a person or entity, in such a way that would cause Microsoft to be considered to operate as a Consumer Reporting Agency under FCRA.

**Trademarks**

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark \\\& Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third\-party trademarks or logos are subject to those third\-party's policies.

**Resources**

[https://ai.azure.com/explore/models/Phi\-3\-small\-128k\-instruct/version/4/registry/azureml?tid\=72f988bf\-86f1\-41af\-91ab\-2d7cd011db47](https://ai.azure.com/explore/models/Phi-3-small-128k-instruct/version/4/registry/azureml?tid=72f988bf-86f1-41af-91ab-2d7cd011db47)