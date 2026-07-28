# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This module link output on run conditionally.

If condition is `true`, link `output` to `generated_batch_train_file_path` and `generated_batch_validation_file_path`.
If condition is `false`, link `output` to `generated_train_file_path` and `generated_validation_file_path`.
"""
import argparse


def copy_file_contents(input_src1, ft_input_train_file_path):
    """
    Copy the contents of one file to another.

    Parameters:
    input_src1 (str): The path to the source file.
    ft_input_train_file_path (str): The path to the destination file.

    Returns:
    None
    """
    # Read the contents of input_src1
    with open(input_src1, "r") as src_file:
        contents = src_file.read()

    # Write the contents to ft_input_train_file_path
    with open(ft_input_train_file_path, "w") as dest_file:
        dest_file.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_batch_train_file_path", type=str)
    parser.add_argument("--generated_batch_validation_file_path", type=str)
    parser.add_argument("--generated_train_file_path", type=str)
    parser.add_argument("--generated_validation_file_path", type=str)
    parser.add_argument("--condition", type=str)
    parser.add_argument("--ft_input_train_file_path", type=str)
    parser.add_argument("--ft_input_validation_file_path", type=str)

    args, _ = parser.parse_known_args()
    print(f"Condition output component received args: {args}.")
    if (
        args.generated_batch_train_file_path is None
        and args.generated_train_file_path is None
    ):
        raise Exception(
            "Got 'generated_batch_train_file_path' and 'generated_train_file_path' both be None."
        )

    condition = args.condition.lower() == "true"
    input_src1 = args.generated_train_file_path
    input_src2 = args.generated_validation_file_path
    ft_input_train_file_path = args.ft_input_train_file_path
    ft_input_validation_file_path = args.ft_input_validation_file_path
    if condition:
        input_src1 = args.generated_batch_train_file_path
        input_src2 = args.generated_batch_validation_file_path
    copy_file_contents(input_src1, ft_input_train_file_path)
    copy_file_contents(input_src2, ft_input_validation_file_path)
