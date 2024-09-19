| 	| |
| -- | -- |
| Label options | Boolean : True if an XPIA (cross domain prompt injected attack) was detected in the response, False otherwise |
| What is this metric? | Detects if the AI system's response contains an XPIA (cross domain prompt injected attack) |
| How does it work? | The XPIA evaluator looks for the presence of XPIA (cross domain prompt injected attacks) in the AI system's response. It returns a label indicating whether or not any was detected, as well as AI-generated reasoning explaining the label choice. |
| When to use it? |	Use it when assessing whether there is XPIA (cross domain prompt injected attacks) in your model's generated responses in real-world applications. |
| What does it need as input? |	This evaluator supports either question/answer or query/response pairs. To use query/response, provide use_qr == "true", a query, and a response. To use question/answer, simply provide a question and an answer. |