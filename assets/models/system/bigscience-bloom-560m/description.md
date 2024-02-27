# **Model Details**

Developed by: BigScience (https://bigscience.huggingface.co/)
Collaborators: Volunteers or individuals with agreements with their employers (further details forthcoming)
Model Type: Transformer-based Language Model
Version: 1.0.0
Languages: Multiple; refer to training data
License: RAIL License v1.0
Release Date Estimate: Monday, 11.July.2022
Funded by:
The French government
Hugging Face
Organizations of contributors (further details forthcoming)

Intended Use:
Created for public research on large language models (LLMs).
Intended for language generation and as a pretrained base model for further fine-tuning.
Use cases include text generation, exploring language model characteristics (e.g., cloze tests, counterfactuals).
Downstream Use:
Tasks leveraging language models such as Information Extraction, Question Answering, and Summarization.
Misuse and Out-of-scope Use:
Out-of-scope uses: High-stakes settings, critical decision-making, biomedical, political, legal, or finance domains.
Misuse: Intentional harm, human rights violations, malicious activities (e.g., spam generation, disinformation).
Intended Users:

Direct Users:

General Public
Researchers
Students
Educators
Engineers/Developers
Non-commercial entities
Community advocates, including human and civil rights groups
Indirect Users:

Users of derivatives created by Direct Users.
Users of Derivatives of the Model as described in the License.
Others Affected (Parties Prenantes):

People and groups referred to by the LLM.
People and groups exposed to LLM outputs or decisions.
People and groups whose original work is included in the LLM.
Bias, Risks, and Limitations:

Foreseeable Harms:
Overrepresentation and underrepresentation of viewpoints.
Presence of stereotypes and personal information.
Generation of hateful, abusive, or violent language; discriminatory or prejudicial language.
Errors, including producing incorrect information as factual.
Recommendations:

Mitigations:
Indirect users should be aware when content is created by the LLM.
Users should be informed of Risks and Limitations, and include age disclaimers or blocking interfaces as necessary.
Pretrained models should have an updated Model Card.
Users should provide mechanisms for affected individuals to provide feedback.
Training Data:

Overview: High-level summary of the training data.
Relevance: For those interested in understanding the basics of what the model has learned.

Note: https://huggingface.co/spaces/bigscience/license

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# **Sample inputs and outputs**

### **Sample input**
```json
{
    "input_data":{
       "input_string":["the meaning of life is"],
       "parameters":{
             "temperature":0.5,
             "top_p":0.5,
             "max_new_tokens":50,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is not always the same for everyone"
  }
]
```
