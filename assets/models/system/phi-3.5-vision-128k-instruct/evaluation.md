In this release, the model enables multi-frame image understanding and reasoning which is based on valuable customer feedback. The hero example multi-frame capabilities include detailed image comparison, multi-image summarization/storytelling and video summarization, which have broad applications in many scenarios. We also observed performance improvement on most single image benchmarks, e.g., boosting MMMU performance from 40.2 to 43.0, MMBench performance from 80.5 to 81.9, document understanding benchmark TextVQA from 70.9 to 72.0. We believe most use cases will benefit from this release, but we encourage users to test the new model in their AI applications. We appreciate the enthusiastic adoption of the Phi-3 model family and continue to welcome all the feedback from the community.

Below are the comparison results on existing multi-image benchmarks. On average, our model outperforms competitor models on the same size and competitive with much bigger models on multi-frame capabilities and video summarization.

BLINK: a benchmark with 14 visual tasks that humans can solve very quickly but are still hard for current multimodal LLMs.

| Benchmark | Phi-3.5-vision-instrust | LlaVA-Interleave-Qwen-7B | InternVL-2-4B | InternVL-2-8B | Gemini-1.5-Flash | GPT-4o-mini | Claude-3.5-Sonnet | Gemini-1.5-Pro | GPT-4o |
|--|--|--|--|--|--|--|--|--|--|
| Art Style | 87.2 | 62.4 | 55.6 | 52.1 | 64.1 | 70.1 | 59.8 | 70.9 | 73.3 |
| Counting | 54.2 | 56.7 | 54.2 | 66.7 | 51.7 | 55.0 | 59.2 | 65.0 | 65.0 |
| Forensic Detection | 92.4 | 31.1 | 40.9 | 34.1 | 54.5 | 38.6 | 67.4 | 60.6 | 75.8 |
| Functional Correspondence | 29.2 | 34.6 | 24.6 | 24.6 | 33.1 | 26.9 | 33.8 | 31.5 | 43.8 |
| IQ Test | 25.3 | 26.7 | 26.0 | 30.7 | 25.3 | 29.3 | 26.0 | 34.0 | 19.3 |
| Jigsaw | 68.0 | 86.0 | 55.3 | 52.7 | 71.3 | 72.7 | 57.3 | 68.0 | 67.3 |
| Multi-View Reasoning | 54.1 | 44.4 | 48.9 | 42.9 | 48.9 | 48.1 | 55.6 | 49.6 | 46.6 |
| Object Localization | 49.2 | 54.9 | 53.3 | 54.1 | 57.3 | 57.4 | 62.3 | 65.6 | 68.0 |
| Relative Depth | 69.4 | 77.4 | 63.7 | 67.7 | 32.8 | 58.1 | 71.8 | 76.6 | 71.0 |
| Relative Reflectance | 37.3 | 34.3 | 32.8 | 38.8 | 32.8 | 27.6 | 36.6 | 38.8 | 40.3 |
| Semantic Correspondence | 36.7 | 31.7 | 31.7 | 22.3 | 32.4 | 31.7 | 45.3 | 48.9 | 54.0 |
| Spatial Relation | 65.7 | 75.5 | 78.3 | 78.3 | 55.9 | 81.1 | 60.1 | 79.0 | 84.6 |
| Visual Correspondence | 53.5 | 40.7 | 34.9 | 33.1 | 29.7 | 52.9 | 72.1 | 81.4 | 86.0 |
| Visual Similarity | 83.0 | 91.9 | 48.1 | 45.2 | 47.4 | 77.8 | 84.4 | 81.5 | 88.1 |
| **Overall** | **57.0** | **53.1** | **45.9** | **45.4** | **45.1** | **51.9** | **56.5** | **61.0** | **63.2** |

Video-MME: comprehensively assess the capabilities of MLLMs in processing video data, covering a wide range of visual domains, temporal durations, and data modalities.

| Benchmark | Phi-3.5-vision-instrust | LlaVA-Interleave-Qwen-7B | InternVL-2-4B | InternVL-2-8B | Gemini-1.5-Flash | GPT-4o-mini | Claude-3.5-Sonnet | Gemini-1.5-Pro | GPT-4o |
|--|--|--|--|--|--|--|--|--|--|
| short (<2min) | 60.8 | 62.3 | 60.7 | 61.7 | 72.2 | 70.1 | 66.3 | 73.3 | 77.7 |
| medium (4-15min) | 47.7 | 47.1 | 46.4 | 49.6 | 62.7 | 59.6 | 54.7 | 61.2 | 68.0 |
| long (30-60) | 43.8 | 41.2 | 42.6 | 46.6 | 52.1 | 53.9 | 46.6 | 53.2 | 59.6 |
| **Overall** | **50.8** | **50.2** | **49.9** | **52.6** | **62.3** | **61.2** | **55.9** | **62.6** | **68.4** |
