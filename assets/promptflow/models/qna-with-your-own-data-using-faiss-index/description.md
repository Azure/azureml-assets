The "QnA with Your Own Data Using Faiss Index" is a Q&A model with GPT3.5 using information from vector search to make the answer more grounded. It involves embedding user's question with LLM, and then using Faiss Index Lookup to find relevant documents based on vectors. By utilizing vector search, users can incorporate their domain knowledge into the prompts, leading to more accurate and relevant answers. The model then uses GPT3.5 to generate an answer to the question with the documents as augmented prompt.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/media/deploy-to-aml-code/sdk/deploy.ipynb" target="_blank">deploy-promptflow-model-python-example</a>|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/go-to-production/deploy-to-aml-code.md" target="_blank">deploy-promptflow-model-cli-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "question": "What's AML SDK V2? Should I use V1 or V2?"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "output": "Using the additional knowledge provided in the prompt, Azure Machine Learning (AML) SDK v2 is a tool that allows you to build a single command or a chain of commands like Python functions. It is useful for constructing workflows, moving from simple to complex concepts incrementally, and managing reusable components in pipelines. It also offers endpoints to streamline model deployments for both real-time and batch inference deployments. In terms of whether to use v1 or v2, it depends on your needs. If you have significant investments in Python SDK v1 and don't need any new features offered by SDK v2, you can continue to use SDK v1. However, you should consider using SDK v2 if you want to use new features like reusable components, managed inferencing, and if you're starting a new workflow or pipeline. All new features and future investments will be introduced in v2. Additionally, SDK v2 offers improved usability, including the ability to compose jobs and pipelines using Python functions and easy evolution from simple to complex tasks. SOURCES: https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2"
    }
}
```