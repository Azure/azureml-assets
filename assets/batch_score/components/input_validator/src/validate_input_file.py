# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch Score Input Validator."""

import argparse
import logging
from typing import List

from batch_api import (
    # BatchApiClient,  # TODO: Uncomment this line to import BatchApiClient
    DataValidationResult
)
from row_validators import (
    BaseValidator,
    JsonValidator,
    SchemaValidator,
    CommonPropertyValidator,
    RowValidationContext,
    RowValidationResult
)
from utils.exceptions import (
    AoaiBatchValidationErrorCode,
    BatchValidationErrorMessage,
    BatchValidationError
)

logger = logging.getLogger(__name__)


def parse_args():
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_file",
        type=str,
        help="Path to the input file",
        required=True
    )

    return parser.parse_args()


def get_row_validators() -> List[BaseValidator]:
    """Get the list of row validators to use for validation."""
    return [
        JsonValidator(),
        SchemaValidator(),
        CommonPropertyValidator()
    ]


def validate_input_file():
    """Validate the input file."""
    try:
        args = parse_args()

        logger.info(f"Reading input data from file '{args.input_data_file}'.")

        with open(args.input_data_file, "r") as f:
            input_data = [line for line in f.readlines() if line.strip()]

        logger.info("Read the input data successfully.")

        row_validators = get_row_validators()

        # batch_api_client = BatchApiClient()  # TODO: Uncomment this line to create an instance of BatchApiClient
        data_validation_result = DataValidationResult()

        if not input_data:
            logger.error("Input file is empty.")

            validation_error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.EMPTY_FILE,
                message=BatchValidationErrorMessage.EMPTY_FILE
            )
            data_validation_result.errors.append(validation_error)

            raise Exception("Input file is empty.")

        logger.info("Starting input validation.")

        for i, line in enumerate(input_data):
            row_validation_context = RowValidationContext(
                raw_input_row=line,
                line_number=i
            )

            for validator in row_validators:
                row_validation_result: RowValidationResult = validator.validate_row(row_validation_context)

                if not row_validation_result.is_success:
                    logger.error(
                        f"Validation failed for input row '{i}'. " +
                        f"Error code: '{row_validation_result.error.code}'. " +
                        f"Error message: '{row_validation_result.error.message}'"
                    )

                    data_validation_result.errors.append(row_validation_result.error)

                    break

        logger.info("Input validation completed successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during input validation: {str(e)}")

        if not data_validation_result:
            data_validation_result = DataValidationResult()

        validation_error = BatchValidationError(
            code=AoaiBatchValidationErrorCode.SERVER_ERROR,
            message=BatchValidationErrorMessage.SERVER_ERROR
        )
        data_validation_result.errors.append(validation_error)

    finally:
        logger.info("Submitting validation result to Batch API.")

        # TODO: Uncomment this line to submit the validation result to Batch API
        # batch_api_client.submit_validation_result(data_validation_result)

        logger.info("Validation result submitted to Batch API. Exiting.")


if __name__ == "__main__":
    validate_input_file()
