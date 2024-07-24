# def
# {\"type\":\"a\",\"element\":1,\"caseInsensitive\":false,\"presentValue\":1.0,\"absentValue\":0.0}
# {\"Equality\":{\"type\":\"a\",\"element\":1,\"caseInsensitive\":false,\"presentValue\":1.0,\"absentValue\":0.0}}

python grade_model.py --predictions_data=artifacts/test.jsonl --ground_truth=artifacts/test.jsonl --ground_truth_column=y_test --prediction_column=y_pred --config_str {\"Equality\":{\"type\":\"a\",\"element\":1,\"caseInsensitive\":false,\"presentValue\":1.0,\"absentValue\":0.0}} --grader_result artifacts.jsonl
python aggregate_predictions.py --data=artifacts.jsonl --config_str {\"Equality\":{\"type\":\"a\",\"element\":1,\"caseInsensitive\":false,\"presentValue\":1.0,\"absentValue\":0.0}} --evaluation_result artifacts


# az ml component create --file ./spec.yaml --workspace training_ws --resource-group training_rg --subscription ed2cab61-14cc-4fb3-ac23-d72609214cfd

# python grade_model.py --predictions_data=test.jsonl --ground_truth=test.jsonl --ground_truth_column=y_test --prediction_column=y_pred --config_str {\"Equality\":{\"type\":\"a\",\"element\":1,\"caseInsensitive\":false,\"presentValue\":1.0,\"absentValue\":0.0}} --grader_result artifacts.jsonl
