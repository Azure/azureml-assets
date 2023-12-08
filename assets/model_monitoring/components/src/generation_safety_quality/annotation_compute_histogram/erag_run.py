def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--metric_names", type=str, required=True)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)

    parser.add_argument("--groundedness_rating_threshold", type=int, default=4)
    parser.add_argument("--similarity_rating_threshold", type=int, default=4)
    parser.add_argument("--relevance_rating_threshold", type=int, default=4)
    parser.add_argument("--fluency_rating_threshold", type=int, default=4)
    parser.add_argument("--coherence_rating_threshold", type=int, default=4)

    parser.add_argument("--prompt_column_name", type=str, default=PROMPT)
    parser.add_argument("--completion_column_name", type=str, default=COMPLETION)
    parser.add_argument("--context_column_name", type=str, default=CONTEXT)
    parser.add_argument("--ground_truth_column_name", type=str, default=GROUND_TRUTH)

    parser.add_argument("--sample_rate", type=float, required=False, default=1.0)
    parser.add_argument(
        "--request_error_rate_threshold",
        type=float,
        default=0.5,
        help="Fail if the running error rate for the endpoint requests "
        "raises above this threshold.",
    )
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    parser.add_argument("--histogram", type=str, required=True)
    parser.add_argument("--samples_index", type=str, required=True)
    parser.add_argument("--groundedness_violations", type=str, required=True)
    parser.add_argument("--fluency_violations", type=str, required=True)
    parser.add_argument("--relevance_violations", type=str, required=True)
    parser.add_argument("--coherence_violations", type=str, required=True)
    parser.add_argument("--similarity_violations", type=str, required=True)

    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    # args = parser.parse_args()
    args = parser.parse_args(args=[
        '--production_dataset', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/521f5fe6-857f-478c-b0cf-18338b21d4d2/joined_data/',
        '--metric_names', 'AcceptableRelevanceScorePerInstance,AggregatedRelevancePassRate',
        '--model_deployment_name', 'gpt-35-turbo-v0301',
        '--workspace_connection_arm_id', '/subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourceGroups/azureml-rag-ci/providers/Microsoft.MachineLearningServices/workspaces/azureml-rag-westus2/connections/azure_open_ai',
        '--prompt_column_name', 'question',
        '--completion_column_name', 'output', '--context_column_name', 'context',
        '--ground_truth_column_name', 'ground_truth', '--sample_rate', '0.1', '--groundedness_rating_threshold', '4',
        '--relevance_rating_threshold', '4', '--similarity_rating_threshold', '4', '--fluency_rating_threshold', '4',
        '--coherence_rating_threshold', '4', '--histogram',
        'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/histogram/',
        '--samples_index', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/samples_index/',
        '--groundedness_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/groundedness_violations/',
        '--fluency_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/fluency_violations/',
        '--similarity_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/similarity_violations/',
        '--coherence_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/coherence_violations/',
        '--relevance_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/relevance_violations/'
    ])

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    request_args["model"] = args.model_deployment_name
    # request_args["n"] = args.num_samples

    token_manager = _WorkspaceConnectionTokenManager(
            connection_name=args.workspace_connection_arm_id,
            auth_header=API_KEY)
    azure_endpoint_domain_name = token_manager.get_endpoint_domain().replace("https://", "")
    azure_openai_api_version = token_manager.get_api_version()

    azure_endpoint_url = _check_and_format_azure_endpoint_url(
        AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
        AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
        azure_endpoint_domain_name,
        azure_openai_api_version,
        args.model_deployment_name  # mdoel
    )
    endpoint_url = azure_endpoint_url

    print(f"Determined endpoint URL {endpoint_url}")

    httpClient = _HTTPClientWithRetry(
        n_retry=args.api_call_retry_max_count,
        backoff_factor=args.api_call_retry_backoff_factor,
    )

    with httpClient.client as session:
        ratings = _query_relevance_score(
            [
                (
                    "What's the highest mountain in the world?",
                    "Mount Everest is the highest mountain in the world. It is located between Nepal and Tibet, "
                    "an autonomous region of China. With an elevation of 29,032 feet (8,849 meters), it is considered "
                    "the tallest point on Earth"
                ),
                (
                    "Should I upgrade to windows11?",
                    "In general, a new prod is better than the old one."
                ),
                (
                    "should I upgrade to windows 11?",
                    "Whether to upgrade to Windows 11 depends on several factors. First, your hardware compatibility, as Windows 11 has specific system requirements. Second, your essential applications are compatible with Windows 11. Third, consider if the new features and interface of Windows 11 appeal to you. Lastly, support for Windows 10 will continue until October 14, 2025."
                )
            ],
            session, endpoint_url, token_manager,
            **request_args,
        )
    print(ratings)
    # calculate_flow_input_output_relevance_score()
    # make two cohorts by good and bad relevance
    # get question, indexed doc relevance from node_run_info.lookup.output[0/1/2].score for the 2 cohorts
    # use t-test to check if they have statistics significant difference
    # if yes:
    # 	generate update index action
    # else:
    # 	generate general action


if __name__ == "__main__":
    run()
