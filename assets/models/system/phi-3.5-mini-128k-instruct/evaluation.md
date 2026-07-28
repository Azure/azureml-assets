### Multilingual
The table below highlights multilingual capability of Phi-3.5-mini on multilingual MMLU, MEGA, and multilingual MMLU-pro datasets. Overall, we observed that even with just 3.8B active parameters, the model is very competitive on multilingual tasks in comparison to other models with a much bigger active parameters.

| Benchmark | Phi-3.5-mini-instruct | Phi-3.0-mini-128k-instruct | Mistral-7B-Instruct-v0.3 | Mistral-Nemo-12B-Ins-2407 | Llama-3.1-8B-Ins | Gemma-2-9B-Ins | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|---|---|---|---|---|---|---|---|---|
| Multilingual MMLU | 55.4 | 51.08 | 47.4 | 58.9 | 56.2 | 63.8 | 77.2 | 72.9 |
| Multilingual MMLU-Pro | 30.9 | 30.21 | 15.0 | 34.0 | 21.4 | 43.0 | 57.9 | 53.2 |
| MGSM | 47.9 | 41.56 | 31.8 | 63.3 | 56.7 | 75.1 | 75.8 | 81.7 |
| MEGA MLQA | 61.7 | 55.5 | 43.9 | 61.2 | 45.2 | 54.4 | 61.6 | 70.0 |
| MEGA TyDi QA | 62.2 | 55.9 | 54.0 | 63.7 | 54.5 | 65.6 | 63.6 | 81.8 |
| MEGA UDPOS | 46.5 | 48.1 | 57.2 | 58.2 | 54.1 | 56.6 | 62.4 | 66.0 |
| MEGA XCOPA | 63.1 | 62.4 | 58.8 | 10.8 | 21.1 | 31.2 | 95.0 | 90.3 |
| MEGA XStoryCloze | 73.5 | 73.6 | 75.5 | 92.3 | 71.0 | 87.0 | 20.7 | 96.6 |
| **Average** | **55.2** | **52.3** | **47.9** | **55.3** | **47.5** | **59.6** | **64.3** | **76.6** |

### Long Context

Phi-3.5-mini supports 128K context length, therefore the model is capable of several long context tasks including long document/meeting summarization, long document QA, long document information retrieval. Phi-3.5-mini outperforms Gemma-2 family which only supports 8K context length and is competitive with other much larger open-weight models such as Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and Mistral-Nemo-12B-Instruct-2407.

| Benchmark | Phi-3.5-mini-instruct | Llama-3.1-8B-instruct | Mistral-7B-instruct-v0.3 | Mistral-Nemo-12B-instruct-2407 | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|---|---|---|---|---|---|---|
| GovReport | 25.9 | 25.1 | 26.0 | 25.6 | 27.8 | 24.8 |
| QMSum | 21.3 | 21.6 | 21.3 | 22.1 | 24.0 | 21.7 |
| Qasper | 41.9 | 37.2 | 31.4 | 30.7 | 43.5 | 39.8 |
| SQuALITY | 25.3 | 26.2 | 25.9 | 25.8 | 23.5 | 23.8 |
| SummScreenFD | 16.0 | 17.6 | 17.5 | 18.2 | 16.3 | 17.0 |
| **Average** | **26.1** | **25.5** | **24.4** | **24.5** | **27.0** | **25.4** |

RULER: a retrieval-based benchmark for long context understanding
| Model | 4K | 8K | 16K | 32K | 64K | 128K | Average |
|--|--|--|--|--|--|--|--|
| **Phi-3.5-mini-instruct** | 94.3 | 91.1 | 90.7 | 87.1 | 78.0 | 63.6 | **84.1** |
| **Llama-3.1-8B-instruct** | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 | **88.3** |
| **Mistral-Nemo-12B-instruct-2407** | 87.8 | 87.2 | 87.7 | 69.0 | 46.8 | 19.0 | **66.2** |

RepoQA: a benchmark for long context code understanding
| Model | Python | C++ | Rust | Java | TypeScript | Average |
|--|--|--|--|--|--|--|
| **Phi-3.5-mini-instruct** | 86 | 67 | 73 | 77 | 82 | **77** |
| **Llama-3.1-8B-instruct** | 80 | 65 | 73 | 76 | 63 | **71** |
| **Mistral-7B-instruct-v0.3** | 61 | 57 | 51 | 61 | 80 | **62** |