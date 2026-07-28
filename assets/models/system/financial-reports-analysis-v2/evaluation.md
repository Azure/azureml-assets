<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `evaluation.md` is highly recommended, but not required. It captures information about the performance of the model. We highly recommend including this section as this information is often used to decide what model to use. -->

The SECQUE Benchmark was developed to evaluate fine\-tuned SLMs in the context of real\-world financial industry applications, specifically targeting their efficacy in assisting financial analysts. It focuses on core scenarios in which SLMs could provide significant value to financial analysts, who analyze information from SEC filings, mainly 10\-K and 10\-Q reports.

The benchmark consists of open\-ended questions designed to reflect real\-world queries posed by financial analysts on SEC filings. Each question entry includes a complex, jargon\-laden query, supporting data, ground\-truth answer, and references to specific sections within the filings. The dataset covers multiple documents and companies, with each question verified to ensure objectivity and dependence solely on the reference data provided, without external information. The questions are categorized into four key task types aligned with common financial analysis activities: risk analysis, ratio analysis, comparative analysis, and insight generation.

The results presented below refer to the second version of the SECQUE benchmark, which contains 565 questions.


| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis\-Phi\-4** | **Phi\-4** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| SECQUE | 69 | 59 | 61 | 66 |

Financial benchmarks (classification):


| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis\-Phi\-4** | **Phi\-4** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| Twitter SA | 88 | 75 | 74 | 78 |
| Twitter Topics | 73 | 59 | 63 | 65 |
| Headline | 89 | 75 | 77 | 80 |
| FPB | 86 | 76 | 79 | 84 |
| **Average F1** | 84 | 71\.3 | 73\.3 | 76\.8 |

Financial benchmarks (exact match)


| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis\-Phi\-4** | **Phi\-4** | **GPT\-4o\-mini** | **GPT\-4o** |
| --- | --- | --- | --- | --- |
| ConvFinQA | 72 | 69 | 72 | 71 |
| FinQA | 68 | 64 | 57 | 67 |
| TACT | 70 | 69 | 68 | 70 |
| **Average exact match** | 70 | 67\.3 | 65\.7 | 69\.3 |

General knowledge benchmarks (comparison with base model):


| **Benchmark** | **Adapted\-AI\-model\-for\-financial\-reports\-analysis\-Phi\-4** | **Phi\-4** | **% Difference** |
| --- | --- | --- | --- |
| TriviaQA | 68 | 68 | 0% |
| MedQA | 73 | 73 | 0% |
| MMLU | 80 | 80 | 0% |
| PIQA | 83 | 82 | 1% |
| WinoGrande | 80 | 81 | \-1% |

\*All evaluations were conducted using temperature 0\.3
