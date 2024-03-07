# Description
This is a 7B parameter, based on Llama2, trained on Hindi, English, and Hinglish. 
More details about the model, its training procedure, and evaluations can be found [here](https://www.sarvam.ai/blog/announcing-openhathi-series).

Note: this is a base model and not meant to be used as is. We recommend first finetuning it on task(s) you are interested in.

The training of the Llama-2-7B model involves three distinct phases:

**Phase 1: Training with Translation**
In this phase, the model's embedding layer is expanded from 32K to 48K tokens, and new token embeddings are randomly initialized. The model is trained on 1.25B tokens of English and Hindi text from the Sangraha corpus, translating paragraphs between the two languages. The goal is for the model to predict the original text given its translation, using a low-rank adapter to avoid affecting existing English skills.

**Phase 2: Training on Bilingual Next Token Prediction**
The next phase involves teaching the model 'world knowledge' in Hindi. Due to the limited availability of Hindi content, English content from sources like Wikipedia is translated using IndicTrans2. Bilingual next token prediction is employed, alternating between Hindi and English sentences. This approach encourages the model to cross-lingually attend to information during next-token prediction. The model is again trained with a low-rank adapter.

**Phase 3: Supervised Fine-tuning**
The base model is fine-tuned on internal datasets and benchmarked on various tasks, including translation, toxicity classification, text simplification, and more. The model is evaluated for English performance against the Llama-2-7B model to check for potential catastrophic forgetting. The use of low-rank adapters and additional vocabulary aims to balance the addition of Hindi skills without drastically affecting English skills.

**Performance Evaluation and Use Cases:**
- The model's translation performance is evaluated against benchmarks like FLoRes-200, showing competitive results in both Devanagari Hindi and Romanized Hindi ↔ English translation.
- Simplified translation is introduced to address formal language output issues. The model's output is compared favorably against GPT-3.5 on a benchmark.
- The model is applied to tasks like content moderation on social media, CoT prompting, and answering questions on a passage. It shows promising results but acknowledges the need for further refinement in certain areas.

**Conclusion and Limitations:**
The presented approach aims to build bilingual LLMs frugally, demonstrating competitiveness with models like GPT-3.5 for Hindi. The scalability of the approach is discussed, suggesting possibilities for increased translation data, diversified language modeling sources, and model adjustments. The limitations include occasional generation of inappropriate content and the need for fine-tuning on specific tasks before practical use. The approach, as presented, provides a foundation for developing domain-specific models for Indic languages.

#### License
OpenHathi-7B-Hi-v0-1-Base is made available under the lamma2 license.

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

## Sample input (for real-time inference)

```json
{
    "input_data": {
        "input_string": [
            "What is meaning of life?"
        ],
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.5,
            "max_new_tokens": 200,
            "do_sample": true
        }
    }
}
```

## Sample output
```json
[
    {
        "0": "What is meaning of life?\n---\nजीवन का अर्थ एक जटिल और बहुआयामी अवधारणा है जो व्यक्तिगत मान्यताओं, मूल्यों और अनुभवों के आधार पर व्यापक रूप से भिन्न होती है। While there is no single answer to this question, many people find meaning in their relationships, their work, their passions, their spirituality, or their contributions to society.\n\nजीवन का अर्थ खोजने के लिए एक व्यक्ति से दूसरे व्यक्ति में बहुत भिन्नता है। Some people find meaning in their family and relationships, while others find meaning in their work or career. कुछ लोग अपने जुनून या रुचियों के माध्यम से अर्थ पाते हैं, जबकि अन्य लोग अपने आध्यात्मिक या धार्मिक विश्वासों के माध्यम से अर्थ पाते हैं।\n\nUltimately, the meaning of life is subjective and can be influenced by a variety of factors, including one's personal experiences, cultural background, and values. जबकि जीवन का कोई एकल अर्थ नहीं है, कई लोग अपने जीवन में अर्थ और उद्देश्य की भावना खोजने के लिए काम करते हैं, जो"
    }
]
```
