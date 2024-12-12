Phi-4 is a state-of-the-art open model built upon a blend of synthetic datasets, data from filtered public domain websites, and acquired academic books and Q&A datasets. The goal of this approach was to ensure that small capable models were trained with data focused on high quality and advanced reasoning.

Phi-4 underwent a rigorous enhancement and alignment process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

For more information, reference the [Phi-4 Technical Report](https://www.microsoft.com/en-us/research/uploads/prod/2024/12/P4TechReport.pdf).

### Model Architecture

Phi-4 is a 14B parameters, dense decoder-only transformer model. 

### Training Data

Our training data is an extension of the data used for Phi-3 and includes a wide variety of sources from:

1. Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code.

2. Newly created synthetic, "textbook-like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.).

3. Acquired academic books and Q&A datasets.

4. High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

Multilingual data constitutes about 8% of our overall data. We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge.
