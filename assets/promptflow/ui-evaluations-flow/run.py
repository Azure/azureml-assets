from promptflow.client import PFClient
from promptflow.evals.evaluators import ViolenceEvaluator as evaluator
from azure.identity import DefaultAzureCredential


def main():

    pf = PFClient()

    data = "samples.json"

    # create run with the flow function and data
    pf.run(
        flow=".",
        data=data,
        column_mapping={
            "ground_truth": "${data.answer}",
            "question": "${data.question}",
            "answer": "${data.answer}",
            "context": "${data.context}",
        },
        stream=False,
    )

if __name__ == "__main__":
    main()
