# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""DataPreprocessor class and runner."""

import json
import re
import jinja2
import subprocess

from azureml._common._error_definition.azureml_error import AzureMLError
from aml_benchmark.utils.exceptions import BenchmarkValidationException, BenchmarkUserException
from aml_benchmark.utils.error_definitions import BenchmarkValidationError, BenchmarkUserError
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files

logger = get_logger(__name__)

jinja2.filters.FILTERS['zip'] = zip
ENV = jinja2.Environment()
ENV.globals.update(zip=zip)


class DatasetPreprocessor(object):
    """DatasetPrerprocessor object class."""

    def __init__(
        self,
        input_dataset: str = None,
        template: str = None,
        user_preprocessor: str = None,
        encoder_config: str = None,
        output_dataset: str = None
    ):
        """Dataset Preprocessor Class.

        :param input_dataset: Path to the jsonl file to load the dataset.
        :param template: A json dictionary where key is the name of the column enclosed in " " and associated \
            dict value is presented using jinja template logic which will be used to extract the \
            respective value from the dataset.
        :param user_preprocessor: Path to the custom preprocessor python script provided by user.
        :param encoder_config: JSON serialized dictionary to perform mapping. Must contain key-value pair \
            "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
            idtolabel or labeltoid mappers. Example format: \
            {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}. This is not applicable to \
            custom scripts.
        :param output_dataset: Path to the jsonl file where the processed data will be saved.
        """
        self.input_dataset = input_dataset
        self.template = template
        self.user_preprocessor = user_preprocessor
        self.encoder_config = encoder_config
        self.output_dataset = output_dataset
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post init call."""
        self.validate()

    def validate(self) -> None:
        """Validate the parameters."""
        if self.input_dataset is None:
            mssg = (
                "Path to jsonl file to load the dataset is not provided."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if len([
            file for file in resolve_io_path(self.input_dataset) if file.endswith(".jsonl")
        ]) == 0:
            mssg = "No .jsonl files found in the given input dataset."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.template is None and self.user_preprocessor is None:
            mssg = (
                "Please provide the input to apply preprocessing logic either via template input or script_path."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.user_preprocessor and not self.user_preprocessor.endswith('.py'):
            mssg = (
                "Please provide python script containing your custom preprocessor logic."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )

    def add_json_filter(self, template) -> str:
        """
        Add "tojson" filter in the template.

        For example: if template_input is
            {"premise":{{premise}}, "hypothesis":{{hypothesis}},"label":{{label}}}
        then, this method returns a formatted template as
            {"premise":{{premise|tojson}}, "hypothesis":{{hypothesis|tojson}},"label":{{label|tojson}}}
        """
        # below pattern is supposed to extract matches within {{}}.
        pattern = r'({{.*?}})'
        regex_pattern = re.compile(pattern)
        matches = re.findall(regex_pattern, template)
        for m in range(len(matches)):
            logger.info(f"{template}, {matches[m]}")
            new_string = '{{'+matches[m].lstrip('{{').rstrip('}}')+'|tojson}}'
            template = template.replace(matches[m], new_string)
        logger.info(f"Final template:{template}")
        return template

    def prep_using_template(self) -> None:
        """Preprocessor run using template."""
        from aml_benchmark.utils.io import resolve_io_path
        data = read_jsonl_files(resolve_io_path(self.input_dataset))
        template = json.dumps(self.template)
        template = json.loads(template)
        template = self.add_json_filter(template)
        env = jinja2.Environment()
        jinja_template = env.from_string(template)
        with open(self.output_dataset, mode='w', encoding='utf8') as f:
            for example in data:
                out = jinja_template.render(example)
                out = out.replace('\'', '\"')
                out_dict = json.loads(out)
                if self.encoder_config:
                    col_to_encode = self.encoder_config.get('column_name')
                    out_dict[col_to_encode] = self.encoder_config.get(str(out_dict.get(col_to_encode)))
                logger.info('Loaded dictionary', out_dict)
                f.write(json.dumps(out_dict) + "\n")
        return

    def run(self) -> None:
        """Preprocessor runner."""
        if self.user_preprocessor:
            self.run_user_preprocessor()
            return
        if self.encoder_config:
            self.encoder_config = json.loads(self.encoder_config)
        if self.template:
            # all preprocessing will be done based on their provided pattern
            self.prep_using_template()
            return

    def run_user_preprocessor(self) -> None:
        """Preprocessor run using custom script."""
        try:
            _ = subprocess.check_output(
                f"python {self.user_preprocessor} --input_path {self.input_dataset} \
                    --output_path {self.output_dataset}",
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            error_message = e.output.strip()
            raise BenchmarkUserException._with_error(
                AzureMLError.create(BenchmarkUserError, error_details=error_message)
            )
