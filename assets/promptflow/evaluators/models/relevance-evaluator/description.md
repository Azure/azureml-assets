## Relevance Evaluator
| Score characteristics	| Score details |
| -- | -- |
| Score range |	Integer [1-5]: where 1 is bad and 5 is good |
| What is this metric? | Measures the extent to which the model's generated responses are pertinent and directly related to the given questions. |
| How does it work? | The relevance measure assesses the ability of answers to capture the key points of the context. High relevance scores signify the AI system's understanding of the input and its capability to produce coherent and contextually appropriate outputs. Conversely, low relevance scores indicate that generated responses might be off-topic, lacking in context, or insufficient in addressing the user's intended queries. |
| When to use it? |	Use the relevance metric when evaluating the AI system's performance in understanding the input and generating contextually appropriate responses. |
| What does it need as input? |	Question, Context, Generated Answer |

Built-in prompt used by Large Language Model judge to score this metric (For question answering data format):

Relevance measures how well the answer addresses the main aspects of the question, based on the context. Consider whether all and only the important aspects are contained in the answer when evaluating relevance. Given the context and question, score the relevance of the answer between one to five stars using the following rating scale: 

> One star: the answer completely lacks relevance 
> 
> Two stars: the answer mostly lacks relevance 
>
> Three stars: the answer is partially relevant 
>
> Four stars: the answer is mostly relevant 
>
> Five stars: the answer has perfect relevance 
>
> This rating value should always be an integer between 1 and 5. So the rating produced should be 1 or 2 or 3 or 4 or 5.