# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Read the args from preprocess component."""

import os
import openai

openai.api_type = "azure"
openai.api_base = "https://daholsteauseaoai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ["OPENAI_API_KEY"]

import json
import argparse
from datetime import datetime


def read_jsonl(input_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get('text')
            data.append(text)
    return data


def distillation_datagen(parsed_args):
    """Datagen."""
    data = read_jsonl(parsed_args.input_data)

    cur_time_str = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")

    print(
        "Writing file: %s" % os.path.join(parsed_args.output_data, "file-" + cur_time_str + ".txt")
    )
    with open(
        os.path.join(parsed_args.output_data, "file-" + cur_time_str + ".txt"), "wt"
    ) as f:
        for item in data:
            message_text = [{"role":"system","content":"You are an AI assistant that summarizes text"},{"role":"user","content":"Write a concise summary of the following: "+item}]
            print
            try:
                completion = openai.ChatCompletion.create(
                    engine="gpt4",
                    messages=message_text,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                ground_truth = completion.choices[0].message.content
                print(ground_truth)

                json.dump({"text": item, "ground_truth": ground_truth}, f)
                f.write('\n')  # Add newline because JSON dump does not do this automatically
                f.flush()
            except Exception as e:
                print(f"An error occurred: {e}")


def main():
    """Parse args and pre process."""
    print("Hello Python World")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_data", type=str)
    args = parser.parse_args()

    distillation_datagen(args)


if __name__ == "__main__":
    main()
