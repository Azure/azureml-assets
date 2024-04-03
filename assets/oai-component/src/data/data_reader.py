from data.base_dao import BaseDAO


def get_data(file_path, y_test_column_name=None, y_pred_column_name=None):
    return BaseDAO(file_path, y_test_column_name, y_pred_column_name)
