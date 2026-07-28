<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `evaluation.md` is highly recommended, but not required. It captures information about the performance of the model. We highly recommend including this section as this information is often used to decide what model to use. -->

We provide two types of benchmark results, the first one is specific to the Trade Compliance domain and based on an in-house evaluation dataset created by subject matter experts. The second one is based on general publicly available datasets that evaluate language understanding, math skill, etc.

## Domain-specific benchmarks
The Trade Compliance evaluation dataset consists of questions from four sub-domains:

1. Tariffs & Duty Management: This section contains questions related to the product code and import duty rate specific to a country/region, (e.g., European Union, USA, etc.), e.g., What is the EU TARIC for a smartphone?
	- We also report the performance of the model on "Complex - Standard" questions, which involve numerical reasoning and/or calculations, e.g., How much import duties would I pay to import 100 kgs of polyethylene containing 4%  alpha-olefin monomers in Singapore?
2. Import/Export Operations: Questions based on the rules and regulations related to import/export, e.g., What is the purpose of the US CTPAT Trade Compliance Program?
3. Trade Compliance (TC) Policy & Oversight: Questions related to Trade Compliance, e.g., Which countries/regions do we consider as EU customs territory?
4. Trade Sanctions & Export Control: Questions related to current market conditions, e.g., How can I use the Commerce Country Chart when I have the ECCN?

Additionally, the evaluation dataset contains "Complex - Combination" questions, which are questions that contain multiple sub-queries and potentially span two or more sub-domains within Trade Compliance, e.g., I am a Germany based automotive manufacturer looking to export EV batteries into the US. What license or document requirements as well as tariff code should I consider for the import?

The scores in the table below are on a scale of 0-2, where 0 is for an incorrect answer, 1 is for a partially correct answer, and 2 is for a fully correct answer.

| Category            | GPT-4o | GPT-4o-mini | Phi-4 | Adapted-AI-model-for-supply-chain-trade-regulations-analysis |
|---------------------|--------|-------------|-------|--------------------------------------|
| Tariffs & Duty Management | 1.57 | 1.51 | 1.49 | 1.55 |
| Import/Export Operations | 1.83 | 1.86 | 1.88 | 1.93 |
| TC Policy & Oversight | 1.84 | 1.86 | 1.87 | 1.91 |
| Trade Sanctions & Export Control | 1.73 | 1.72 | 1.75 | 1.79 |
| Complex - Standard | 1.65 | 1.58 | 1.50 | 1.65 |
| Complex - Combination | 1.46 | 1.45 | 1.46 | 1.58 |
| **Average Score** | 1.64 | 1.62 | 1.61 | 1.67 |

## General benchmarks
| Category | Benchmark | Phi-4 | Adapted-AI-model-for-supply-chain-trade-regulations-analysis |
| -------- | --------- | ------------------- | ----------------------- |
| Popular aggregated benchmark | AGI Eval | 48.60 | 48.47 |
|                              | MMLU | 76.95 | 76.86 |
| Language Understanding       | HellaSwag | 63.17 | 62.51 |
| Reasoning                    | ARC Easy | 81.48 | 81.36 |
|                              | ARC Challenge | 55.89 | 54.44 |
|                              | WinoGrande | 76.95 | 77.43 |
| Math                         | GSM-8K (strict match) | 90.07 | 89.01 |
|                              | GSM-8K (flexible extract) | 90.14 | 89.01 |