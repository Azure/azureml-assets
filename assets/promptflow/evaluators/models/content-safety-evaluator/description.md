| 	| |
| -- | -- |
| Score range |	Integer [0-7]: where 0 is the least harmful and 7 is the most harmful. A text label inis also provided. |
| What is this metric? | Measures comprehensively the severity level of the content harm of a response, covering violence, sexual, self-harm, and hate and unfairness as 4 harmful categories. |
| How does it work? | The Content Safety evaluator leverages AI-assisted evaluators including `ViolenceEvaluator`, `SexualEvaluator`, `SelfHarmEvaluator`, `HateUnfairnessEvaluator` with a language model as a judge on the response to a user query. See the [definitions and severity scale](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in?tabs=severity#risk-and-safety-evaluators) for these AI-assisted evaluators. |
| When to use it? |	Use it when assessing the readability and user-friendliness of your model's generated responses in real-world applications. |
| What does it need as input? |	Query, Response |