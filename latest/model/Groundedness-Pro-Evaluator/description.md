|   | |
| -- | -- |
| Score range | Boolean: [true, false]: false if response is ungrounded and true if it's grounded.  |
| What is this metric? | Groundedness Pro (powered by Azure AI Content Safety) detects whether the generated text response is consistent or accurate with respect to the given context in a retrieval-augmented generation question and answering scenario. It checks whether the response adheres closely to the context in order to answer the query, avoiding speculation or fabrication, and outputs a true/false label. |
| How does it work? | Groundedness Pro leverages an Azure AI Content Safety Service custom language model fine-tuned to a natural language processing task called Natural Language Inference (NLI), which evaluates claims in response to a query as being entailed or not entailed by the given context. |
| When to use it? | The recommended scenario is retrieval-augmented generation question and answering (RAG QA). Use the Groundedness Pro metric when you need to verify that AI-generated responses align with and are validated by the provided context. It's essential for applications where contextual accuracy is key, like information retrieval and question and answering. This metric ensures that the AI-generated answers are well-supported by the context. |
| What does it need as input? | Query, Context, Response |
