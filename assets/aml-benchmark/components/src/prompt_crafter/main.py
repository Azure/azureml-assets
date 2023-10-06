from argparse import ArgumentParser
import logging
from runner import run_prompt_crafter, PromptChoices

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=PromptChoices)
    parser.add_argument("--n_shots", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=0, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_mltable", type=str, required=True)
    parser.add_argument("--few_shot_dir", type=str, required=False)
    parser.add_argument("--input_filename", type=str, required=False)
    parser.add_argument("--few_shot_filename", type=str, required=False)
    parser.add_argument("--metadata_keys", type=str, required=False)
    parser.add_argument("--prompt_pattern", type=str, required=False)
    parser.add_argument("--few_shot_pattern", type=str, required=False)
    parser.add_argument("--few_shot_separator", type=str, required=False)
    parser.add_argument("--prefix", type=str, required=False)
    parser.add_argument("--label_map", type=str, required=False)
    parser.add_argument("--system_message", type=str, required=False)
    parser.add_argument("--output_pattern", type=str, required=False)
    parser.add_argument("--additional_payload", type=str, required=False)

    return parser.parse_args()

def main():
    args = parse_args()
    prompt_crafter = PromptCrafter(**vars(args))
    prompt_crafter.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
