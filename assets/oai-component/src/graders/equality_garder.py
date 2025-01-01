import pandas as pd
from graders.base_grader import Grader


class EqualityGrader(Grader):

    def compute(self, pred_data, ground_truth):
        y_test, y_pred = ground_truth.get_y_test(), pred_data.get_y_pred()
        if self.config.caseInsensitive:
            y_test = [x.lower() for x in y_test]
            y_pred = [x.lower() for x in y_pred]
        return pd.DataFrame({"Equality": [self.config.presentValue if x == y else self.config.absentValue for x, y in zip(y_test, y_pred)]})
