import pandas as pd


class BaseDAO:

    def __init__(self, file_path, y_test_column_name, y_pred_column_name):
        self.file_path = file_path
        self.y_test_column_name = y_test_column_name
        self.y_pred_column_name = y_pred_column_name
        self.data = pd.read_json(self.file_path, lines=True, dtype=False)

    def get_y_test(self):
        y_test_column_name = self.y_test_column_name if self.y_test_column_name is not None else self.data.columns[0]
        return self.data[y_test_column_name]

    def get_y_pred(self):
        y_pred_column_name = self.y_pred_column_name if self.y_pred_column_name is not None else self.data.columns[1]
        return self.data[y_pred_column_name]

    def get_values(self, column_name):
        return self.data[column_name]
