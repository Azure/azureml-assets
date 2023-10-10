def extract_using_template(self, key: str=None) -> None:
        # result_df: pd.DataFrame, key: str = None, processor_order: List = None
        """Postprocessor run using template."""
        import pdb;pdb.set_trace();
        result_df = pd.DataFrame()
        result_df = self.read_ground_truth_dataset(result_df)
        # read the predcition dataset
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        # if self.prediction_column_name in predicted_data[0].keys():
        #     key = self.prediction_column_name
        # else:
        #     key = "0"
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        template = self.template
        env = jinja2.Environment()
        jinja_template = env.from_string(template)
        for row in predicted_data:
            if key != self.prediction_column_name:
                row[self.prediction_column_name] = row.get(key)
            predicted = row.get(self.prediction_column_name)
            if isinstance(predicted, list):
                try:
                    out_string = jinja_template.render(predicted)
                    pred_list.append(out_string)
                except Exception as e:
                    error_msg = "jinja2.exceptions.UndefinedError: 'list object' has no attribute 'split'"
                    if error_msg in e:
                        curr_pred_list = []
                        for i in range(0, len(predicted)):
                            curr_pred = {self.prediction_column_name:predicted[i]}
                            out_string = jinja_template.render(curr_pred)
                            curr_pred_list.append(out_string)
                        pred_list.append(curr_pred_list)
                    else:
                        raise BenchmarkUserException._with_error(
                            AzureMLError.create(BenchmarkUserError, error_details=e)
                        )
            else:
                out_string = jinja_template.render(predicted)
                pred_list.append(out_string)
        if isinstance(pred_list[0], list) and len(pred_list[0])>1:
            cols = [f"{self.prediction_column_name}_{i+1}" for i in range(len(pred_list[0]))] 
        else:
            cols = self.prediction_column_name
        # result_df[self.prediction_column_name] = pred_list
        result_df[cols] = pred_list
        result_df = self.read_pred_probs_dataset(result_df)
        # combine the records in one pandas dataframe and write it to the jsonl file.
        result_df.to_json(self.result, lines=True, orient='records')
        return