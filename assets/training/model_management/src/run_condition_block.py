import argparse
import json
import logging

# Assuming swallow_all_exceptions and get_logger are defined elsewhere
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions
from azureml.model.mgmt.utils.logging_utils import get_logger

logger = get_logger(__name__)

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True, help="Condition")
    parser.add_argument("--input-args", type=str, required=True, help="Input args")
    parser.add_argument("--result", type=str, required=True, help="Output result file path")
    return parser

@swallow_all_exceptions(logger)
def run():
    """Run preprocess."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    condition = args.condition
    input_args = args.input_args
    result_path = args.result
    input_args = json.loads(input_args)
    logger.info(f"Run preprocess with input args: {input_args}, and condition: {condition}")

    try:
        # Use eval to evaluate the condition with a context dictionary
        result = eval(condition, input_args)
    except Exception as e:
        logger.error(f"Error evaluating condition: {e}")
        result = False

    logger.info(f"Condition result: {result}")

    # Write the result to the output file
    with open(result_path, 'w') as f:
        f.write(str(result))

if __name__ == "__main__":
    run()
