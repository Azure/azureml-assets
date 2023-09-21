# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""DataPreprocessor class and runner."""

import json
import os
import re
import jinja2
import sys
from utils.exceptions import BenchmarkValidationException
from utils.error_definitions import BenchmarkValidationError
from utils.logging import get_logger
from utils.io import read_jsonl_files
from azureml._common._error_definition.azureml_error import AzureMLError

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
sys.path.append(parent_folder)
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

        :param input_dataset: Path to the directory to load the dataset.
        :param template: A json dictionary where key is the name of the column enclosed in " " and associated \
            dict value is presented using jinja template logic which will be used to extract the \
            respective value from the dataset.
        :param user_preprocessor: Path to the custom preprocessor python script provided by user.
        :param encoder_config: Dictionary in json format that contains column/key name to encode associated \
            with dict key `column_name` followed by idtolabel or labeltoid mappers as key-value pair. Example format: \
            {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}. This is not applicable to \
            custom scripts.
        :param output_dataset: Path to the dump the processed .jsonl file.
        """
        self.input_dataset = input_dataset
        self.template = template
        self.user_preprocessor = user_preprocessor
        self.encoder_config = encoder_config
        self.output_dataset = output_dataset

    def __post_init__(self) -> None:
        """Post init call."""
        self._validate()

    def validate(self) -> None:
        """Validate the parameters."""
        if self.input_dataset is None:
            mssg = (
                "Path to load the dataset is not provided."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.template is None and self.user_preprocessor is None:
            mssg = (
                "Please provide the input to apply preprocessing logic either via template input or \
                script_path."
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
        """Add tojson filter in the template."""
        patt = r'({{.*?}})'
        pat = re.compile(patt)
        matches = re.findall(pat, template)
        for m in range(len(matches)):
            logger.info(f"{template}, {matches[m]}")
            s1 = '{{'+matches[m].lstrip('{{').rstrip('}}')+'|tojson}}'
            template = template.replace(matches[m], s1)
        logger.info(f"Final template:{template}")
        return template

    def prep_using_template(self) -> None:
        """Preprocessor run using template."""
        data = read_jsonl_files([self.input_dataset])
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
        """Prerpocessor run using custom template."""
        try:
            os.system(
                f'python {self.user_preprocessor} --input_path {self.input_dataset} \
                --output_path {self.output_dataset}'
            )
        except Exception as e:
            logger.exception('Script failed', e)
        return
