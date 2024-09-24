| | |
| -- | -- |
| Score range | Float [0-1] |
| What is this metric? | Measures how closely the generated text matches a reference text based on n-gram overlap. |
| How does it work? | The BLEU score calculates the geometric mean of the precision of n-grams between the model-generated text and the reference text, with an added brevity penalty for shorter generated text. The precision is computed for unigrams, bigrams, trigrams, etc., depending on the desired BLEU score level. The more n-grams that are shared between the generated and reference texts, the higher the BLEU score. |
| When to use it? | Use the BLEU score when you want to evaluate the similarity between the generated text and reference text, especially in tasks such as machine translation or text summarization, where n-gram overlap is a significant indicator of quality. |
| What does it need as input? | Ground Truth Response, Generated Response |
