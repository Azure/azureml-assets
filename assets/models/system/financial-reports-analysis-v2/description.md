<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

The **Adapted AI model for financial reports analysis** (Phi\-4, preview) is a state\-of\-the\-art small language model (SLM) based on the Phi\-4 architecture, designed specifically for analyzing financial reports. It has been fine\-tuned on a few hundred million tokens derived from instruction data over financial documents, including SEC filings (10\-K, 10\-Q, 8\-K reports) and mathematical reasoning tasks. The model is optimized to handle complex financial language and to understand data contained in tables, making it suitable for SEC report analysis, including data extraction, summarization, and common financial formulas. It can also perform more complex reasoning tasks, such as comparing companies and identifying trends across different time periods.

The adapted AI model for financial reports analysis (Phi\-4) is a dense, decoder\-only transformer model with 14B parameters, optimized for financial reports analysis. It supports a 16K context length, making it capable of processing long financial documents and providing coherent, context\-aware completions. The model is fine\-tuned with supervised fine\-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

*NOTE: This model is in preview*
