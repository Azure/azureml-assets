| 	| |
| -- | -- |
| Score range |	Integer [1-5]: 1 is the lowest quality and 5 is the highest quality. |
| What is this metric? | Retrieval measures the quality of search without ground truth. It focuses on how relevant the context chunks (encoded as a string) are to address a query and how the most relevant context chunks are surfaced at the top of the list. |
| How does it work? | The retrieval metric is calculated by instructing a language model to follow the definition (in the description) and a set of grading rubrics, evaluate the user inputs, and output a score on a 5-point scale (higher means better quality). Learn more about our [definition and grading rubrics](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in?tabs=warning#ai-assisted-retrieval). |
| When to use it? |	The recommended scenario is the quality of search in information retrieval and retrieval augmented generation, when you don't have ground truth for chunk retrieval rankings. Use the retrieval score when you want to assess to what extent the context chunks retrieved are highly relevant and ranked at the top for answering your users' queries. |
| What does it need as input? |	Query, Context |
