from promptflow.client import PFClient


def main():

    pf = PFClient()

    data = "samples.json"
    # create run with the flow function and data
    base_run = pf.run(
        flow=".",
        data=data,
        column_mapping={
            "ground_truth": "${data.answer}",
            "question": "${data.question}",
            "answer": "${data.answer}",
            "context": "${data.context}",
        },
        stream=True,
    )

if __name__ == "__main__":
    main()
