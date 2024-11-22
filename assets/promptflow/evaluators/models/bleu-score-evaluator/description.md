| | |
| -- | -- |
| Score range | Float [0-1]: higher means better quality. |
| What is this metric? | BLEU (Bilingual Evaluation Understudy) score is commonly used in natural language processing (NLP) and machine translation. It measures how closely the generated text matches the reference text. |
| How does it work? | The BLEU score calculates the geometric mean of the precision of n-grams between the model-generated text and the reference text, with an added brevity penalty for shorter generated text. The precision is computed for unigrams, bigrams, trigrams, etc., depending on the desired BLEU score level. The more n-grams that are shared between the generated and reference texts, the higher the BLEU score. |
| When to use it? | The recommended scenario is Natural Language Processing (NLP) tasks. It's widely used in text summarization and text generation use cases. |
| What does it need as input? | Response, Ground Truth |
