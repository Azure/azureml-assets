| 	| |
| -- | -- |
| Score range |	Float [0-1]: higher means better quality. |
| What is this metric? | F1 score measures the similarity by shared tokens between the generated text and the ground truth, focusing on both precision and recall. |
| How does it work? | The F1-score computes the ratio of the number of shared words between the model generation and the ground truth. Ratio is computed over the individual words in the generated response against those in the ground truth answer. The number of shared words between the generation and the truth is the basis of the F1 score: precision is the ratio of the number of shared words to the total number of words in the generation, and recall is the ratio of the number of shared words to the total number of words in the ground truth. |
| When to use it? |	The recommended scenario is Natural Language Processing (NLP) tasks. Use the F1 score when you want a single comprehensive metric that combines both recall and precision in your model's responses. It provides a balanced evaluation of your model's performance in terms of capturing accurate information in the response. |
| What does it need as input? | Response, Ground Truth |
