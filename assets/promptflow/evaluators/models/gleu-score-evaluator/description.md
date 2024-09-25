| | |
| -- | -- |
| Score range | Float [0-1] |
| What is this metric? | Measures the degree of overlap between the generated text and both the reference text and source text, balancing between precision and recall. |
| How does it work? | The GLEU score is computed by averaging the precision and recall of n-grams between the generated text and both the reference text and source text. It considers both the overlap of n-grams with the reference (similar to BLEU) and penalizes for over-generation. The score provides a balanced metric, where a value of 1 represents perfect overlap, and 0 represents no overlap. |
| When to use it? | Use the GLEU score when you want a more balanced evaluation of generated text that considers both the precision and recall of n-gram overlap, especially useful in evaluating machine translation or paraphrasing tasks. |
| What does it need as input? | Ground Truth Response, Generated Response |
