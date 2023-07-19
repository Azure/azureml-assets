# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains unit tests for the Numerical Data Drift Metrics.
It compares the drift measures of metrics implemented in Spark against their Scipy implementation 
"""

from data_drift_compute_metrics.numerical_data_drift_metrics import compute_numerical_data_drift_measures_tests
from shared_utilities.io_utils import init_spark
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial import distance
import pandas as pd
import pyspark.sql as pyspark_sql
import pytest
import numpy as np
import unittest
import math

distance_measures = [
    "NormalizedWassersteinDistance",
    "JensenShannonDistance",
    "PopulationStabilityIndex",
    "TwoSampleKolmogorovSmirnovTest",
]


test_cases = [
    {"name": "1a",
     "scenario": "Distance between identical distributions should be exaclty 0",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "1b",
     "scenario": "Distance between two samples of the different sizes of the same distributions should be close to 0",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 10_000,
    },
    {"name": "1c",
     "scenario": "Non-zero distance between distributions with the same mean and sample size, but differnt std_dev",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 10,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "2a",
    "scenario": "Confirming distance of moderately drifted distribution against Scipy",
     "x_mean": 50,
     "y_mean": 53,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "2b",
    "scenario": "Weak distance value dependence on sample size. Distance should be close to test 2a.",
     "x_mean": 50,
     "y_mean": 53,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 10_000,
    },
    {"name": "2c",
    "scenario": "Decreasing std_dev in prodcution dataset. Increase in distance expected when compared to test 2a.",
     "x_mean": 50,
     "y_mean": 53,
     "x_std_dev": 15,
     "y_std_dev": 5,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "2d",
    "scenario": "Decreasing both sample size and std_dev in prodcution dataset. Similar distance as test 2c expected.",
     "x_mean": 50,
     "y_mean": 53,
     "x_std_dev": 15,
     "y_std_dev": 5,
     "x_obs": 100_000,
     "y_obs": 10_000,
    },
    {"name": "3a",
    "scenario": "Large drift. Bound distnace measures (Jensen-Shannon) approaching maximum theoretical values.",
     "x_mean": 50,
     "y_mean": 100,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "3b",
    "scenario": "Extreme drift scenario. Bound distnace measures (Jensen-Shannon) should reach their maximum theoretical value.",
     "x_mean": 50,
     "y_mean": 250,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 100_000,
     "y_obs": 100_000,
    },
    {"name": "4a",
    "scenario": "Extreme drift scenario. All values the same, but different in baseline and prod.",
     "x_mean": 50,
     "y_mean": 51,
     "x_std_dev": 0,
     "y_std_dev": 0,
     "x_obs": 9_000,
     "y_obs": 9_000,
    },
    {"name": "5a",
     "scenario": "Distance between identical distributions should be exaclty 0",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 9_000,
     "y_obs": 9_000,
    },
    {"name": "5b",
     "scenario": "Distance between two samples of the different sizes of the same distributions should be close to 0",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 9_000,
     "y_obs": 900,
    },
    {"name": "5c",
     "scenario": "Non-zero distance between distributions with the same mean and sample size, but differnt std_dev",
     "x_mean": 50,
     "y_mean": 50,
     "x_std_dev": 15,
     "y_std_dev": 10,
     "x_obs": 9_000,
     "y_obs": 9_000,
    },
    {"name": "6a",
    "scenario": "Confirming distance of moderately drifted distribution against Scipy",
     "x_mean": 50,
     "y_mean": 51,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 9_000,
     "y_obs": 9_000,
    },
    {"name": "6b",
    "scenario": "Weak distance value dependence on sample size. Distance should be close to test 6a.",
     "x_mean": 50,
     "y_mean": 51,
     "x_std_dev": 15,
     "y_std_dev": 15,
     "x_obs": 9_000,
     "y_obs": 900,
    },
    {"name": "6c",
    "scenario": "Decreasing std_dev in prodcution dataset. Increase in distance expected when compared to test 6a.",
     "x_mean": 50,
     "y_mean": 51,
     "x_std_dev": 15,
     "y_std_dev": 5,
     "x_obs": 9_000,
     "y_obs": 9_000,
    },
    {"name": "6d",
    "scenario": "Decreasing both sample size and std_dev in prodcution dataset. Similar distance as test 6c expected.",
     "x_mean": 50,
     "y_mean": 51,
     "x_std_dev": 15,
     "y_std_dev": 5,
     "x_obs": 9_000,
     "y_obs": 900,
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

    def create_spark_df(self, numerical_data):
        column_values=['column1']
        pd_df = pd.DataFrame(data=numerical_data, columns=column_values)
        df = self.spark.createDataFrame(pd_df)
        return df

    def create_test_distributions(self, dist_params: dict):
        np.random.seed(0)
        x = np.random.normal(dist_params["x_mean"], dist_params["x_std_dev"], dist_params["x_obs"])
        np.random.seed(0)
        y = np.random.normal(dist_params["y_mean"], dist_params["y_std_dev"], dist_params["y_obs"])
        return x,y


    def compute_optimal_histogram_bin_edges(self, x_array,y_array):
    
        #find overall min and max values for both distributions
        min_value = min(np.amin(x_array), np.amin(y_array))
        max_value = max(np.amax(x_array), np.amax(y_array))
    
        #add overall min and max value to both distributions in order estimate the appropriate number of bins
        #for both samples that span the entire range of data.
        x_array_extended = np.append(x_array,[min_value, max_value])
        y_array_extended = np.append(y_array,[min_value, max_value])    

        #select the amount of bins used. The smaller amount of bins is recommended to ensure both x_array and y_array have well populated histograms 
        if min_value == max_value:
            number_of_bins = 1
            bin_edges = np.array([min_value/1.01, min_value*1.01])
    
        else:
            #compute optimal bin edges for both distributions after having been extended to [min_value,max_value]
            x_bin_edges=np.histogram_bin_edges(x_array_extended, bins='sturges')
            y_bin_edges=np.histogram_bin_edges(y_array_extended, bins='sturges')
            number_of_bins = min(len(x_bin_edges), len(y_bin_edges))
            bin_edges=np.linspace(min_value, max_value, number_of_bins)

        return bin_edges


    # # # EXPECTED NORMALIZED WASSERSTEIN DISTANCE # # # 

    def normed_wasserstein_distance_numerical(self, x_array, y_array):
        norm = max(np.std(x_array),0.001)
        return wasserstein_distance(x_array, y_array) / norm

    # # # EXPECTED JENSEN-SHANNON DISTANCE # # # 

    def jensen_shannon_distance_numerical(self, x_array, y_array):
        bin_edges = self.compute_optimal_histogram_bin_edges(x_array, y_array)
        x_percent = np.histogram(x_array, bins=bin_edges)[0] / len(x_array)
        y_percent = np.histogram(y_array, bins=bin_edges)[0] / len(y_array)

        return distance.jensenshannon(x_percent, y_percent, base=2)

    # # # EXPECTED PSI # # #
    
    def psi_numerical(self, x_array, y_array):
        bin_edges = self.compute_optimal_histogram_bin_edges(x_array, y_array)

        x_count = np.histogram(x_array, bins=bin_edges)[0]
        y_count = np.histogram(y_array, bins=bin_edges)[0] 
        
        # Laplace smoothing (incrementing the count of each bin by 1)
        # to avoid zero values in bins and have the SPI value be finit
        x_count = x_count + 1
        y_count = y_count + 1 

        x_percent = x_count / len(x_array) 
        y_percent = y_count / len(y_array)   

        psi = 0.0
        for i in range(len(x_percent)):
            psi += (y_percent[i] - x_percent[i]) * np.log(y_percent[i] / x_percent[i])

        return psi

    # # # EXPECTED TWO SAMPLE KS TEST # # # 

    def two_sample_ks_test_numerical(self, x_array, y_array):
        return ks_2samp(x_array, y_array).pvalue
    
    # # # MAIN TEST FUNCTION # # # 

    def test_compute_numerical_data_drift_metrics(self):
        """Test compute distance measures for numerical metrics."""
        column_values = ['column1']
        numerical_threshold = 0.1

        for distance_measure in distance_measures:
            print(f'###########################################')
            print(f'TESTING DISTANCE MEASURE: {distance_measure}')
            print(f'###########################################')

            for test_case in test_cases:
                x,y = self.create_test_distributions(test_case)

                if distance_measure == "NormalizedWassersteinDistance":
                    expected_distance = self.normed_wasserstein_distance_numerical(x,y)
                elif distance_measure == "JensenShannonDistance":
                    expected_distance = self.jensen_shannon_distance_numerical(x,y)
                elif distance_measure == "PopulationStabilityIndex":
                    expected_distance = self.psi_numerical(x,y)
                elif distance_measure == "TwoSampleKolmogorovSmirnovTest":
                    expected_distance = self.two_sample_ks_test_numerical(x,y)    
                else:
                    raise ValueError(f"Distance measure {distance_measure} not in {distance_measures}")

                x_df = self.create_spark_df(x)
                y_df = self.create_spark_df(y)
                x_count = x_df.count()
                y_count = y_df.count()

                #Excluding KS-tess with more than 10k samples
                if not (distance_measure == "TwoSampleKolmogorovSmirnovTest" 
                    and (x_count >= 10_000  or y_count >= 10_000)):
                    
                    output_df = compute_numerical_data_drift_measures_tests(
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
                    print((
                        f'expected value:{self.round_to_n_significant_digits(expected_distance,4)},' 
                        f'measured value:{self.round_to_n_significant_digits(metric_value,4)}'
                        ))

                    if(abs(expected_distance) < 1e-6):
                        self.assertAlmostEqual(metric_value, 0, 6)
                    else:
                        self.assertAlmostEqual(float(expected_distance)/metric_value, 1.0, 1)