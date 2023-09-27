# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
This module link output on run conditionally.

If condition is `true`, link `output` to `input_a`.
If condition is `false`, link `output` to `input_b`.
"""
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_a", type=str)
parser.add_argument("--input_b", type=str)
#parser.add_argument("--condition", type=str)
parser.add_argument("--output", type=str)
args, _ = parser.parse_known_args()
print(f"Condition output component received args: {args}.")
if args.input_a is None and args.input_b is None:
    raise Exception("Got 'input_a' and 'input_b' both be None.")

#condition = args.condition.lower() == "true"

if args.input_b is None:
    print("Copy input_a to output")
    shutil.copytree(args.input_a, args.output)
elif args.input_a is None:
    print("Copy input_b to output")
    shutil.copytree(args.input_b, args.output)
else:
    raise Exception("Got 'input_a' and 'input_b' both be None.")

