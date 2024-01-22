from ...batch_pool.quota.quota_client import QuotaClient
from ...utils.common import str2bool
from ..common_enums import ApiType


# TODO: Name this ARGUMENT_SPECIFICATION
COMMAND_LINE_ARGUMENT_SPECIFICATION = {
    # TODO: headers with booleans fail during session.post.
    #  Prevent users from providing additional_headers that json.loads with boolean values.
    '--additional_headers': {
        'const': None,
        'default': None,
        'help': 'A serialized JSON string of headers to be included with every scoring request.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--additional_properties': {
        'const': None,
        'default': None,
        'help': 'A serialized JSON string of properties to be included in the payload for every scoring request.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--configuration_file': {
        'const': None,
        'default': None,
        'help': 'JSON file containing configuration values for the batch score component. '
                'Values in this file will override the parameter values.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--amlbi_async_mode': {
        'dest': 'async_mode',
        'default': False,
        'help': 'If enabled, the component processes multiple mini-batches within each process on the compute.'
                ' If disabled, the component processes a single mini-batch at a time.',
        'required': False,
        'type': str2bool,
    },
    '--api_key_name': {
        'default': None,
        'help': 'The name of the secret that contains the API key.'
                " Required when using the authentication_type 'api_key'.",
        'required': False,
        'type': str,
    },
    '--api_type': {
        'choices': [
            ApiType.Completion,
            ApiType.ChatCompletion,
            ApiType.Embedding,
            ApiType.Vesta,
            ApiType.VestaChatCompletion,
        ],
        'default': ApiType.Completion,
        'help': 'The API used for scoring.',
        'required': False,
        'type': str,
    },
    '--app_insights_connection_string': {
        'const': None,
        'default': None,
        'help': 'The connection string for an Application Insights instance.'
                ' If provided, debug logs are sent to this Application Insights instance.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--app_insights_log_level': {
        'choices': [
            'debug',
            'info',
            'warning',
            'error',
            'critical',
        ],
        'const': None,
        'default': 'debug',
        'help': 'Minimum log level to emit to Application Insights.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--authentication_type': {
        'choices': [
            'api_key',
            'azureml_workspace_connection',
            'managed_identity',
        ],
        'default': 'managed_identity',
        'help': 'The type of authentication used for scoring.',
        'required': False,
        'type': str,
    },
    '--batch_pool': {
        'const': None,
        'default': None,
        'help': 'The name of the batch pool.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--batch_size_per_request': {
        'default': 1,
        'help': 'The number of rows to score against the model in a single HTTP request.'
                ' Only supported for the Embeddings API.'
                ' Must be between 1 and 2000.',
        'required': False,
        'type': int,
    },
    '--connection_name': {
        'default': None,
        'help': 'The name of the connection that contains the API key to use for scoring.'
                ' This connection must belong to the same workspace as the compute.'
                " Required when using the authentication_type 'azureml_workspace_connection.' ",
        'required': False,
        'type': str,
    },
    '--debug_mode': {
        'default': False,
        'help': 'If enabled, debug-level (and above) logs are emitted to App Insights.'
                ' If disabled, info-level (and above) logs are emitted to App Insights.',
        'required': False,
        'type': str2bool,
    },
    '--ensure_ascii': {
        'default': False,
        'help': 'Ensure ASCII.',
        'required': False,
        'type': str2bool,
    },
    '--image_input_folder': {
        'const': None,
        'default': None,
        'help': 'Image input folder.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--initial_worker_count': {
        'default': 5,
        'help': 'Initial number of workers.',
        'required': False,
        'type': int,
    },
    '--max_retry_time_interval': {
        'const': None,
        'default': None,
        'help': 'The maximum duration for which to attempt to score any single request.',
        'nargs': '?',
        'required': False,
        'type': int,
    },
    '--max_worker_count': {
        'default': 200,
        'help': 'Maximum number of workers.',
        'required': False,
        'type': int,
    },
    '--mini_batch_results_out_directory': {
        'default': None,
        # TODO: add validation to ensure that this parameter is not empty when save_mini_batch_results is enabled.
        'help': 'The name of the directory where the results of individual minibatches are stored.'
                ' Required when `save_mini_batch_results` is enabled.',
        'required': False,
        'type': str,
    },
    '--online_endpoint_url': {
        'default': None,
        'help': 'Online endpoint URL.',
        'required': False,
        'type': str,
    },
    '--output_behavior': {
        # TODO: add the choices 'append_row' and 'summary_only'.
        'default': None,
        'help': 'If set to `append_row`, the output of each scoring request is appended to the single, potentially'
                ' large output file. If set to `summary_only`, the output file contains only the summary of the'
                ' scoring run. Use the `save_mini_batch_results` parameter to save the results'
                ' of the individual minibatches.',
        'required': False,
        'type': str,
    },
    '--quota_audience': {
        'default': None,
        'help': 'Quota audience.',
        'required': False,
        'type': str,
    },
    '--quota_estimator': {
        'choices': QuotaClient.ESTIMATORS.keys(),
        'default': 'completion',
        'help': 'Quota estimator.',
        'required': False,
        'type': str,
    },
    '--request_path': {
        'help': 'Request path.',
        'required': False,
        'type': str,
    },
    '--save_mini_batch_results': {
        'default': None,
        'help': 'If enabled, the results of individual minibatches are stored in the directory'
                ' specified by `mini_batch_results_out_directory`.',
        'required': False,
        'type': str,
    },
    '--scoring_url': {
        'default': None,
        'help': 'Scoring URL.',
        'required': False,
        'type': str,
    },
    '--segment_large_requests': {
        # TODO: convert this to a boolean
        'choices': [
            'enabled',
            'disabled',
        ],
        'help': 'Segment large requests.',
        'required': False,
        'type': str,
    },
    '--segment_max_token_size': {
        'default': 800,
        'help': 'Maximum token size.',
        'required': False,
        'type': int,
    },
    '--service_namespace': {
        'default': None,
        'help': 'Service namespace.',
        'required': False,
        'type': str,
    },
    '--stdout_log_level': {
        'choices': [
            'debug',
            'info',
            'warning',
            'error',
            'critical',
        ],
        'const': None,
        'default': 'debug',
        'help': 'Minimum log level to emit to job stdout.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--tally_exclusions': {
        'choices': [
            'none',
            'all',
            'errors_only',
        ],
        'default': 'none',
        'help': 'Tally exclusions.',
        'required': False,
        'type': str,
    },
    '--tally_failed_requests': {
        'default': False,
        'help': 'Tally failed requests.',
        'required': False,
        'type': str2bool,
    },
    '--token_file_path': {
        'default': None,
        'help': 'Path to the token file. This is only required for local runs.',
        'required': False,
        'type': str,
    },
    '--user_agent_segment': {
        'const': None,
        'default': None,
        'help': 'User agent segment.',
        'nargs': '?',
        'required': False,
        'type': str,
    },
}

COMMAND_LINE_ARGUMENT_SPECIFICATION_FOR_FILE_CONFIGURATION = {
    '--amlbi_async_mode': {
        'dest': 'async_mode',
        'default': False,
        'help': 'If enabled, the component processes multiple mini-batches within each process on the compute.'
                ' If disabled, the component processes a single mini-batch at a time.',
        'required': False,
        'type': str2bool,
    },
    '--configuration_file': {
        'const': None,
        'default': None,
        'help': 'JSON file containing configuration values for the batch score component. ',
        'nargs': '?',
        'required': False,
        'type': str,
    },
    '--partitioned_scoring_results': {
        'default': None,
        'help': 'The name of the directory where the partitioned scoring results will be stored.'
                ' Required when `save_partitioned_scoring_results` is true.',
        'required': False,
        'type': str,
    },
}
