# **Model Details**
Note: Use of this model is governed by the Meta license. Click on View License above.
Code Llama family of large language models (LLMs).

Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This is the repository for the base 7B version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding. Links to other models can be found in the index at the bottom.

**Model Developers** Meta AI

**Variations** Code Llama comes in three model sizes, and three variants:

Code Llama: base models designed for general code synthesis and understanding
Code Llama - Python: designed specifically for Python
Code Llama - Instruct: for instruction following and safer deployment

All variants are available in sizes of 7B, 13B and 34B parameters.

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** Code Llama is an auto-regressive language model that uses an optimized transformer architecture.

**Model Dates** Code Llama and its variants have been trained between January 2023 and July 2023.

**Status** This is a static model trained on an offline dataset. Future versions of Code Llama - Instruct will be released as we improve model safety with community feedback.

**License** A custom commercial license is available at: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

# **Intended Use**
**Intended Use Cases** Code Llama and its variants is intended for commercial and research use in English and relevant programming languages. The base model Code Llama can be adapted for a variety of code synthesis and understanding tasks, Code Llama - Python is designed specifically to handle the Python programming language, and Code Llama - Instruct is intended to be safer to use for code assistant and generation applications.

**Out-of-scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Code Llama and its variants.

# **Hardware and Software**
**Training Factors** We used custom training libraries. The training and fine-tuning of the released models have been performed Meta’s Research Super Cluster.

**Carbon Footprint**  In aggregate, training all 9 Code Llama models required 400K GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 65.3 tCO2eq, 100% of which were offset by Meta’s sustainability program.

# **Training Data**
All experiments reported here and the released models have been trained and fine-tuned using the same data as Llama 2 with different weights.

# **Evaluation Results**

See evaluations for the main models and detailed ablations in Section 3 and safety evaluations in Section 4 of the research paper.

# **Ethical Considerations and Limitations**
Code Llama and its variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Code Llama’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Code Llama, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available at [https://ai.meta.com/llama/responsible-use-guide/](https://ai.meta.com/llama/responsible-use-guide)

# Model Evaluation samples

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## **Sample inputs and outputs (for real-time inference)**

### **Sample input**
```json
{
    "input_data": {
        "input_string": ["Develop a Python function to sort a list of integers in ascending order"], 
        "parameters": { 
            "return_full_text": false,
            "do_sample":true
        }
    }
}
```

### **Sample output**
```json
[
    {
        "0": "def sort_integers(int_list):\n    \"\"\"This function takes an a list of integers and sorts it in ascending order using build in `sorted` function which returns\n    a new sorted list.\n\n    Args:\n        int_list (list): list of integers to be sorted.\n    Returns:\n        \"list\": sorted list\n\n    \"\"\"\n    return sorted(int_list)\nlst = [3, 7, -4, 2, 1]\nlst1 = [15, 2, 1, 8, 4, 3]\nprint(\"unsorted list\", lst)\nprint(\"sorted list\", sort_integers(lst))\nprint(\"sorted list\", sort_integers(lst1))\n"
    }
]
```
