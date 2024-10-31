|   | |
| -- | -- |
| Score range | Integer [1-5]: where 1 is bad and 5 is good |
| What is this metric? | Uses service-based evaluation to measure how well the model's generated answers align with information from the source data (user-defined context). |
| How does it work? | The groundedness measure calls Responsible AI service to assess the correspondence between claims in an AI-generated answer and the source context, making sure that these claims are substantiated by the context. Even if the responses from LLM are factually correct, they'll be considered ungrounded if they can't be verified against the provided sources (such as your input source or your database). |
| When to use it? | Use the groundedness metric when you need to verify that AI-generated responses align with and are validated by the provided context. It's essential for applications where factual correctness and contextual accuracy are key, like information retrieval, question-answering, and content summarization. This metric ensures that the AI-generated answers are well-supported by the context. |
| What does it need as input? | Query, Context, Generated Response |