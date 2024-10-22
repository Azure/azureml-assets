| | |
| -- | -- |
| Score range | Float [0-1] |
| What is this metric? | Measures the quality of the generated text by comparing it to a reference text using n-gram recall, precision, and F1-score. |
| How does it work? | The ROUGE score (Recall-Oriented Understudy for Gisting Evaluation) evaluates the similarity between the generated text and reference text based on n-gram overlap, including ROUGE-N (unigram, bigram, etc.), and ROUGE-L (longest common subsequence). It calculates precision, recall, and F1 scores to capture how well the generated text matches the reference text. Rouge type options are "rouge1" (Unigram overlap), "rouge2" (Bigram overlap), "rouge3" (Trigram overlap),  "rouge4" (4-gram overlap), "rouge5" (5-gram overlap), "rougeL" (L-graph overlap) |
| When to use it? | Use the ROUGE score when you need a robust evaluation metric for text summarization, machine translation, and other natural language processing tasks, especially when focusing on recall and the ability to capture relevant information from the reference text. |
| What does it need as input? | Rouge type, Ground Truth Response, Generated Response |