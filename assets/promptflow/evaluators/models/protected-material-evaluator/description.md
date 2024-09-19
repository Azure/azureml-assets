| 	| |
| -- | -- |
| Label options | Boolean : True if protected material was detected in the response, False otherwise |
| What is this metric? | Detects if the AI system's response contains protected material |
| How does it work? | The protected material evaluator looks for the presence of protected material in the AI system's response. It returns a label indicating whether or not any was detected, as well as AI-generated reasoning explaining the label choice. |
| When to use it? |	Use it when assessing whether there is protected material in your model's generated responses in real-world applications. |
| What does it need as input? |	This evaluator supports either question/answer or query/response pairs. To use query/response, provide use_qr == "true", a query, and a response. To use question/answer, simply provide a question and an answer. |