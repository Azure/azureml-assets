| 	| |
| -- | -- |
| Score range |	Integer [1-5]: where 1 is bad and 5 is good |
| What is this metric? | Measures the extent to which retrieved information is relevant to the user's queries and contexts |
| How does it work? | The retrieval measure assesses the relevance and effectiveness of the retrieved context chunks with respect to the query. High retrieval scores indicate that the AI system has successfully extracted and ranked the most relevant information at the top, without introducing bias from external knowledge and ignoring factual correctness. Conversely, low retrieval scores suggest that the AI system has failed to surface the most relevant context chunks at the top of the list and/or introduced bias and ignored factual correctness. |
| When to use it? |	Use the retrieval metric when evaluating the AI system's performance in retrieving information for additional context (e.g. a RAG scenario). |
| What does it need as input? |	Query, Context |