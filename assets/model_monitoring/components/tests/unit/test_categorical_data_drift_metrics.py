# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains unit tests for the Categorical Data Drift Metrics.
It compares the drift measures of metrics implemented in Spark against their SciPy implementation 
"""

from data_drift_compute_metrics.categorical_data_drift_metrics import compute_categorical_data_drift_measures_tests
from shared_utilities.io_utils import init_spark
from scipy.spatial import distance
from scipy.stats import chisquare
import pandas as pd
import pyspark.sql as pyspark_sql
import pytest
import numpy as np
import unittest
import math
import random

distance_measures = [
    "JensenShannonDistance",
    "PopulationStabilityIndex",
    "PearsonsChiSquaredTest",
]

# Sample categorical distributions
s_a = 'a'*100 + 'b'*100 + 'c'*100 + 'd'*100 + 'e'*100 + 'f'*100 + 'g'*100 + 'h'*100
s_b = 'a'*97  + 'b'*105 + 'c'*99  + 'd'*98  + 'e'*101 + 'f'*102 + 'g'*97  + 'h'*103  
s_c = 'a'*10  + 'b'*11  + 'c'*9   + 'd'*10  + 'e'*10  + 'f'*10  + 'g'*13  + 'h'*10  
s_d = 'a'*180 + 'b'*80  + 'c'*70  + 'd'*170 + 'e'*200 + 'f'*10  + 'g'*130 + 'h'*100  
s_e = 'a'*100 + 'b'*100 + 'c'*100 + 'd'*100 + 'e'*100 + 'f'*100 + 'g'*100 + 'h'*1  
s_f = 'a'*100 + 'b'*100 + 'c'*100 + 'f'*100 + 'g'*100 + 'h'*1  
s_g = 'i'*1   + 'j'*1   + 'k'*2  
s_h = 'a'*100 + 'b'*100  
s_i = 'a'*34  + 'b'*34  
s_j = 'a'*92  + 'b'*23  
s_k = 'c'*100 + 'd'*100  

test_cases = [
    {"name": "1a",
     "scenario": "Distance between identical distributions should be exactly 0.",
     "baseline_col": s_b,
     "production_col": s_b,
    },
    {"name": "1b",
     "scenario": "Minimal drift. Both distributions contain the same categories.",
     "baseline_col": s_a,
     "production_col": s_b,
    },
    {"name": "1c",
     "scenario": "Marginal drift. Both distributions contain the same categories.",
     "baseline_col": s_a,
     "production_col": s_c,
    },
    {"name": "1d",
     "scenario": "Drifted distributions. Both distributions contain the same categories.",
     "baseline_col": s_a,
     "production_col": s_d,
    },
    {"name": "1e",
     "scenario": "Drift caused by a single category. Both distributions contain the same categories.",
     "baseline_col": s_a,
     "production_col": s_e,
    },
    {"name": "1f",
     "scenario": "Drifted distributions. Production dataset is missing some categories.",
     "baseline_col": s_a,
     "production_col": s_f,
    },
    {"name": "1g",
     "scenario": "Distributions with no categories in common. Bound distance measures (Jensen-Shannon) should reach their maximum theoretical value.",
     "baseline_col": s_a,
     "production_col": s_g,
    },
    {"name": "2a",
     "scenario": "Dual value identical distributions with the same sample size. Distance measures should be 0.",
     "baseline_col": s_h,
     "production_col": s_h,
    },
    {"name": "2b",
     "scenario": "Dual value identical distributions with different sample sizes. Distance measures should be 0.",
     "baseline_col": s_h,
     "production_col": s_i,
    },
    {"name": "2c",
     "scenario": "Drifted Dual value distributions.",
     "baseline_col": s_h,
     "production_col": s_j,
    },
    {"name": "2d",
     "scenario": "Dual value distributions with different values. Bound distance measures (Jensen-Shannon) should reach their maximum theoretical value.",
     "baseline_col": s_h,
     "production_col": s_k,
    },
]


@pytest.mark.unit
class TestComputeDataDriftMetrics(unittest.TestCase):
    """Test class for data drift compute metrics component component and utilities."""
    
    def __init__(self, *args, **kwargs):  
        super(TestComputeDataDriftMetrics, self).__init__(*args, **kwargs)  
        self.spark = init_spark()  

    def round_to_n_significant_digits(self, number, n):  
        if number == 0:  
            return 0  
        else:  
            return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)  
       
    def get_metric_value(self, df: pyspark_sql.DataFrame, metric_name: str):
        """Get metric value of the first row of a given column from a dataframe."""
        #df.show()
        return df.filter(f"metric_name = '{metric_name}'").first().metric_value

    def create_spark_df(self, categorical_data):
        column_values=['column1']
        pd_df = pd.DataFrame(data=categorical_data, columns=column_values)
        df = self.spark.createDataFrame(pd_df)
        return df

    def create_categorical_data_from_string(self, sample_string: str):
        cat_list = [letter for letter in sample_string]
        random.shuffle(cat_list)
        return cat_list


    # # # EXPECTED JENSEN-SHANNON DISTANCE # # # 

    def jensen_shannon_distance_categorical(self, x_list, y_list):
    
        # unique values observed in x and y
        values = set(x_list + y_list)
        
        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])
    
        x_ratios = x_counts / np.sum(x_counts)  #Optional as JS-D normalizes probability vectors
        y_ratios = y_counts / np.sum(y_counts)

        return distance.jensenshannon(x_ratios, y_ratios, base=2)

    # # # EXPECTED PSI # # #
    
    def psi_categorical(self, x_list, y_list):
        
        # unique values observed in x and y
        values = set(x_list + y_list)
        
        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])

        # Laplace smoothing (incrementing the count of each bin by 1)
        # to avoid zero values in bins and have the SPI value be finit
        x_counts = x_counts + 1
        y_counts = y_counts + 1

        x_ratios = x_counts / np.sum(x_counts)
        y_ratios = y_counts / np.sum(y_counts)

        psi = 0
        for i in range(len(x_ratios)):
            psi += (y_ratios[i] - x_ratios[i]) * np.log(y_ratios[i] / x_ratios[i])
    
        return psi

    # # # EXPECTED CHI SQUARED TEST # # # 

    def chi_squared_test_categorical(self, x_list, y_list):
        values = set(x_list + y_list)
        
        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])

        x_ratios = x_counts / np.sum(x_counts)
        expected_y_counts = x_ratios * len(y_list)

        return chisquare(y_counts, expected_y_counts).pvalue
    
    # # # MAIN TEST FUNCTION # # # 

    def test_compute_categorical_data_drift_metrics(self):
        """Test compute distance measures for categorical metrics."""
        column_values = ['column1']
        numerical_threshold = 0.1

        for distance_measure in distance_measures:
            print(f'###########################################')
            print(f'TESTING DISTANCE MEASURE: {distance_measure}')
            print(f'###########################################')

            for test_case in test_cases:
                x = self.create_categorical_data_from_string(test_case['baseline_col'])
                y = self.create_categorical_data_from_string(test_case['production_col'])

                if distance_measure == "JensenShannonDistance":
                    expected_distance = self.jensen_shannon_distance_categorical(x,y)
                elif distance_measure == "PopulationStabilityIndex":
                    expected_distance = self.psi_categorical(x,y)
                elif distance_measure == "PearsonsChiSquaredTest":
                    expected_distance = self.chi_squared_test_categorical(x,y)    
                else:
                    raise ValueError(f"Distance measure {distance_measure} not in {distance_measures}")

                x_df = self.create_spark_df(x)
                y_df = self.create_spark_df(y)
                x_count = x_df.count()
                y_count = y_df.count()


                    
                output_df = compute_categorical_data_drift_measures_tests(
                    x_df,
                    y_df,
                    x_count,
                    y_count,
                    distance_measure,
                    column_values,
                    numerical_threshold)

                metric_value = self.get_metric_value(output_df, distance_measure)
            
                print('-------------------------')
                print(f'test: {test_case["name"]}')
                print(f'test scenario: {test_case["scenario"]}')
                print(f'expected value: {self.round_to_n_significant_digits(expected_distance,4)}') 
                print(f'measured value: {self.round_to_n_significant_digits(metric_value,4)}')

                if(abs(expected_distance) < 1e-6):
                    self.assertAlmostEqual(metric_value, 0, 6)
                else:
                    self.assertAlmostEqual(float(expected_distance)/metric_value, 1.0, 1)