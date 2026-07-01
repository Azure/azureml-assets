| 	| |
| -- | -- |
| Score range |	Integer [1-5]: 1 is the lowest quality and 5 is the highest quality. |
| What is this metric? | Similarity measures the degrees of similarity between the generated text and its ground truth with respect to a query. |
| How does it work? | The similarity metric is calculated by instructing a language model to follow the definition (in the description) and a set of grading rubrics, evaluate the user inputs, and output a score on a 5-point scale (higher means better quality). Learn more about our [definition and grading rubrics](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in?tabs=warning#ai-assisted-similarity). |
| When to use it? |	The recommended scenario is NLP tasks with a user query. Use it when you want an objective evaluation of an AI model's performance, particularly in text generation tasks where you have access to ground truth responses. Similarity enables you to assess the generated text's semantic alignment with the desired content, helping to gauge the model's quality and accuracy.|
| What does it need as input? |	Response, Ground Truth |
