<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `note.md` is highly recommended, but not required. It captures information about how your model is created. We highly recommend including this section to provide transparency for the customers. -->

## Intended Use

### Primary Use Cases

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

### Out-of-Scope Use Cases

The model is not specifically designed or evaluated for all downstream purposes. The model is not designed to provide any financial advice or recommendations. Developers should be mindful of common limitations of language models when selecting use cases and ensure that they evaluate and mitigate potential issues related to accuracy, safety, and fairness before deploying the model, particularly in high\-risk scenarios. Additionally, developers should adhere to relevant laws and regulations (including privacy and trade compliance laws) applicable to their use cases.

**Nothing written here should be interpreted as or deemed a restriction or modification to the license the model is released under.**

## Responsible AI Considerations

Adapted\-AI\-model\-for\-financial\-reports\-analysis\-Phi\-4, like other language models, can exhibit biases or generate outputs that may not be suitable for all contexts. Developers should be aware of the following considerations:

* Quality of Service: The adapted AI model for financial reports analysis (Phi\-4) model is trained primarily on English text. Languages other than English do not perform as well. English language varieties with less representation in the training data might not perform as well as standard American English.
* Representation of Harms \& Perpetuation of Stereotypes: This model can over\- or under\-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post\-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real\-world patterns and societal biases.
* Inappropriate or Offensive Content: This model may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
* Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (for example: privacy, trade, etc.). Important areas for consideration include:

* Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (for example: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
* High\-Risk Scenarios: Developers should assess the suitability of using models in high\-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (for example: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
* Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end\-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use\-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
* Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
* Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

Prompts and completions are passed through a default configuration of Azure AI Content Safety classification models to detect and prevent the output of harmful content. Learn more about [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview). Configuration options for content filtering vary when you deploy a model for production in Azure AI; [learn more](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview).

## Training Data

The adapted AI model for financial reports analysis (Phi\-4) was fine\-tuned on a highly specialized dataset totaling a few hundred million tokens, including:

* SEC filings, including 10\-K, 10\-Q, and 8\-K reports
* Textbook\-like materials focusing on finance
* Verified common questions and answer pairs on SEC reports, scaled synthetically
* Mathematical reasoning tasks

The training data was carefully selected to ensure that the model excels at financial reasoning, risk assessment, and understanding complex financial terms and scenarios.

## Hardware

Note that by default, the Phi\-4 model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA H100

## Inference samples

Our endpoint has two supported URIs: `score` (used in the `Test` tab in the AI-Foundry) and `chat/completions`

The content contains the system prompt (### Instruction), reference data (### Document) and a financial question (### Question).
We recommend using the following system prompt:

```Instruction:\nYou are given a financial question and a a financial document. Your task is to answer the question based on the document.\n```

*Sample input 1 - score*
```json
{
    "input_data": {
        "input_string": [{"role": "user", "content":  "### Instruction:\nYou are given a financial question and a a financial document. Your task is to answer the question based on the document.\n### Document:\n\nCOCA COLA CO 10-K form for the fiscal year ended 2023-12-31, page 26: production capabilities, disrupt our supply chain or impact demand for our products. Increasing concern over climate change also may result in additional legal or regulatory requirements designed to reduce or mitigate the effects of carbon dioxide and other greenhouse gas emissions on the environment, and\/or may result in increased disclosure obligations. Increased energy or compliance costs and expenses due to increased legal or regulatory requirements may cause disruptions in, or an increase in the costs associated with, the manufacturing and distribution of our beverage products. The physical effects and transition costs of climate change and legal, regulatory or market initiatives to address climate change could have a long-term adverse impact on our business and results of operations. In addition, from time to time we establish and publicly announce goals and targets to reduce the Coca-Cola system's carbon footprint by increasing our use of recycled packaging materials, expanding our renewable energy usage, and participating in environmental and sustainability programs and initiatives organized or sponsored by nongovernmental organizations and other groups to reduce greenhouse gas emissions industrywide. If we and our bottling partners fail to achieve or improperly report on our progress toward achieving our carbon footprint reduction goals and targets, the resulting negative publicity could adversely affect consumer preference for our beverage products. \nAdverse weather conditions could reduce the demand for our products.\nThe sales of our products are influenced to some extent by weather conditions in the markets in which we operate. Unusually cold or rainy weather during the summer months may have a temporary effect on the demand for our products and contribute to lower sales, which could have an adverse effect on our results of operations for such periods. \n### Question:\n\nHow could adverse weather conditions affect The Coca-Cola Company's sales and profitability?\n### Response:\n"}],
        "parameters": {"temperature": 0.3, "max_new_tokens": 4096}
    }
}
```

*Sample input 2 - score*
```json
{
    "input_data": {
        "input_string": [{"role": "user", "content":  "### Instruction:\nYou are given a financial question and a a financial document. Your task is to answer the question based on the document.\n### Document:\n\nNVIDIA CORP 10-K form for the fiscal year ended 2024-01-28, page 50: \nNVIDIA Corporation and Subsidiaries\n\n\nConsolidated Statements of Income\n\n(In millions, except per share data)|    |                                                        |              |        |              |        |              |        |        |    |        |\n|---:|:-------------------------------------------------------|:-------------|:-------|:-------------|:-------|:-------------|:-------|:-------|:---|:-------|\n|  1 |                                                        | Year Ended   |        |              |        |              |        |        |    |        |\n|  2 |                                                        | Jan 28, 2024 |        | Jan 29, 2023 |        | Jan 30, 2022 |        |        |    |        |\n|  3 | Revenue                                                | $            | 60,922 |              |        | $            | 26,974 |        | $  | 26,914 |\n|  4 | Cost of revenue                                        | 16,621       |        |              | 11,618 |              |        | 9,439  |    |        |\n|  5 | Gross profit                                           | 44,301       |        |              | 15,356 |              |        | 17,475 |    |        |\n|  6 | Operating expenses                                     |              |        |              |        |              |        |        |    |        |\n|  7 | Research and development                               | 8,675        |        |              | 7,339  |              |        | 5,268  |    |        |\n|  8 | Sales, general and administrative                      | 2,654        |        |              | 2,440  |              |        | 2,166  |    |        |\n|  9 | Acquisition termination cost                           | -            |        |              | 1,353  |              |        | -      |    |        |\n| 10 | Total operating expenses                               | 11,329       |        |              | 11,132 |              |        | 7,434  |    |        |\n| 11 | Operating income                                       | 32,972       |        |              | 4,224  |              |        | 10,041 |    |        |\n| 12 | Interest income                                        | 866          |        |              | 267    |              |        | 29     |    |        |\n| 13 | Interest expense                                       | (257)        |        |              | (262)  |              |        | (236)  |    |        |\n| 14 | Other, net                                             | 237          |        |              | (48)   |              |        | 107    |    |        |\n| 15 | Other income (expense), net                            | 846          |        |              | (43)   |              |        | (100)  |    |        |\n| 16 | Income before income tax                               | 33,818       |        |              | 4,181  |              |        | 9,941  |    |        |\n| 17 | Income tax expense (benefit)                           | 4,058        |        |              | (187)  |              |        | 189    |    |        |\n| 18 | Net income                                             | $            | 29,760 |              |        | $            | 4,368  |        | $  | 9,752  |\n| 20 | Net income per share:                                  |              |        |              |        |              |        |        |    |        |\n| 21 | Basic                                                  | $            | 12.05  |              |        | $            | 1.76   |        | $  | 3.91   |\n| 22 | Diluted                                                | $            | 11.93  |              |        | $            | 1.74   |        | $  | 3.85   |\n| 24 | Weighted average shares used in per share computation: |              |        |              |        |              |        |        |    |        |\n| 25 | Basic                                                  | 2,469        |        |              | 2,487  |              |        | 2,496  |    |        |\n| 26 | Diluted                                                | 2,494        |        |              | 2,507  |              |        | 2,535  |    |        |\nSee accompanying notes to the consolidated financial statements.### Question:\n\nWhat is the Operating Profit Margin for NVIDIA in 2024, and how does it compare to 2023?\n### Response:\n"}],
        "parameters": {"temperature": 0.3, "max_new_tokens": 4096}
    }
}
```

*Sample input 1 - chat/completions*
```json
{
    "messages": [
        {
            "role": "user",
            "content": "### Instruction:\nYou are given a financial question and a a financial document. Your task is to answer the question based on the document.\n### Document:\n\nCOCA COLA CO 10-K form for the fiscal year ended 2023-12-31, page 26: production capabilities, disrupt our supply chain or impact demand for our products. Increasing concern over climate change also may result in additional legal or regulatory requirements designed to reduce or mitigate the effects of carbon dioxide and other greenhouse gas emissions on the environment, and\/or may result in increased disclosure obligations. Increased energy or compliance costs and expenses due to increased legal or regulatory requirements may cause disruptions in, or an increase in the costs associated with, the manufacturing and distribution of our beverage products. The physical effects and transition costs of climate change and legal, regulatory or market initiatives to address climate change could have a long-term adverse impact on our business and results of operations. In addition, from time to time we establish and publicly announce goals and targets to reduce the Coca-Cola system's carbon footprint by increasing our use of recycled packaging materials, expanding our renewable energy usage, and participating in environmental and sustainability programs and initiatives organized or sponsored by nongovernmental organizations and other groups to reduce greenhouse gas emissions industrywide. If we and our bottling partners fail to achieve or improperly report on our progress toward achieving our carbon footprint reduction goals and targets, the resulting negative publicity could adversely affect consumer preference for our beverage products. \nAdverse weather conditions could reduce the demand for our products.\nThe sales of our products are influenced to some extent by weather conditions in the markets in which we operate. Unusually cold or rainy weather during the summer months may have a temporary effect on the demand for our products and contribute to lower sales, which could have an adverse effect on our results of operations for such periods. \n### Question:\n\nHow could adverse weather conditions affect The Coca-Cola Company's sales and profitability?\n### Response:\n"
        }
    ],
    "temperature": 0.3,
    "response_format": { "type": "text" }
}
```

*Sample input 2 - chat/completions*
```json
{
    "messages": [
        {
            "role": "user",
            "content":  "### Instruction:\nYou are given a financial question and a a financial document. Your task is to answer the question based on the document.\n### Document:\n\nNVIDIA CORP 10-K form for the fiscal year ended 2024-01-28, page 50: \nNVIDIA Corporation and Subsidiaries\n\n\nConsolidated Statements of Income\n\n(In millions, except per share data)|    |                                                        |              |        |              |        |              |        |        |    |        |\n|---:|:-------------------------------------------------------|:-------------|:-------|:-------------|:-------|:-------------|:-------|:-------|:---|:-------|\n|  1 |                                                        | Year Ended   |        |              |        |              |        |        |    |        |\n|  2 |                                                        | Jan 28, 2024 |        | Jan 29, 2023 |        | Jan 30, 2022 |        |        |    |        |\n|  3 | Revenue                                                | $            | 60,922 |              |        | $            | 26,974 |        | $  | 26,914 |\n|  4 | Cost of revenue                                        | 16,621       |        |              | 11,618 |              |        | 9,439  |    |        |\n|  5 | Gross profit                                           | 44,301       |        |              | 15,356 |              |        | 17,475 |    |        |\n|  6 | Operating expenses                                     |              |        |              |        |              |        |        |    |        |\n|  7 | Research and development                               | 8,675        |        |              | 7,339  |              |        | 5,268  |    |        |\n|  8 | Sales, general and administrative                      | 2,654        |        |              | 2,440  |              |        | 2,166  |    |        |\n|  9 | Acquisition termination cost                           | -            |        |              | 1,353  |              |        | -      |    |        |\n| 10 | Total operating expenses                               | 11,329       |        |              | 11,132 |              |        | 7,434  |    |        |\n| 11 | Operating income                                       | 32,972       |        |              | 4,224  |              |        | 10,041 |    |        |\n| 12 | Interest income                                        | 866          |        |              | 267    |              |        | 29     |    |        |\n| 13 | Interest expense                                       | (257)        |        |              | (262)  |              |        | (236)  |    |        |\n| 14 | Other, net                                             | 237          |        |              | (48)   |              |        | 107    |    |        |\n| 15 | Other income (expense), net                            | 846          |        |              | (43)   |              |        | (100)  |    |        |\n| 16 | Income before income tax                               | 33,818       |        |              | 4,181  |              |        | 9,941  |    |        |\n| 17 | Income tax expense (benefit)                           | 4,058        |        |              | (187)  |              |        | 189    |    |        |\n| 18 | Net income                                             | $            | 29,760 |              |        | $            | 4,368  |        | $  | 9,752  |\n| 20 | Net income per share:                                  |              |        |              |        |              |        |        |    |        |\n| 21 | Basic                                                  | $            | 12.05  |              |        | $            | 1.76   |        | $  | 3.91   |\n| 22 | Diluted                                                | $            | 11.93  |              |        | $            | 1.74   |        | $  | 3.85   |\n| 24 | Weighted average shares used in per share computation: |              |        |              |        |              |        |        |    |        |\n| 25 | Basic                                                  | 2,469        |        |              | 2,487  |              |        | 2,496  |    |        |\n| 26 | Diluted                                                | 2,494        |        |              | 2,507  |              |        | 2,535  |    |        |\nSee accompanying notes to the consolidated financial statements.### Question:\n\nWhat is the Operating Profit Margin for NVIDIA in 2024, and how does it compare to 2023?\n### Response:\n"
        }
    ],
    "temperature": 0.3,
    "response_format": { "type": "text" }
}
```

## Disclaimer

Customer agrees that any information or output resulting from the use of the Adapted\-AI\-model\-for\-financial\-reports\-analysis is for informational or internal process management purposes only, and does not constitute legal, financial, tax planning, or other advice from Microsoft. Customer agrees that it is responsible for its own financial research and financial decisions, and that the solutions and resulting information provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis will not serve as the primary basis for its financial decisions. Customer agrees that Microsoft is not responsible or liable for any decisions or actions customer, or its authorized third parties, take based on information Customer produces or generates as a user of the Adapted\-AI\-model\-for\-financial\-reports\-analysis. No solutions provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis constitute an offer, solicitation of an offer, or advice to buy or sell securities, or any financial instrument or investment by Microsoft.

Customer may not use any of the features or information provided through the Adapted\-AI\-model\-for\-financial\-reports\-analysis as a factor in establishing the financial standing, including the eligibility for credit, hire, insurance, housing, employment or other eligibility or entitlement (including for any other use constituting a permissible purpose under the U.S. Federal Fair Credit Reporting Act (“FCRA”)) of a person or entity, in such a way that would cause Microsoft to be considered to operate as a Consumer Reporting Agency under FCRA.

## License

The model is licensed under the MIT license.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third\-party trademarks or logos are subject to those third\-party's policies.

## Resources

[https://ai.azure.com/explore/models/Phi\-4/version/7/registry/azureml?tid\=72f988bf\-86f1\-41af\-91ab\-2d7cd011db47](https://ai.azure.com/explore/models/Phi-4/version/7/registry/azureml?tid=72f988bf-86f1-41af-91ab-2d7cd011db47)


