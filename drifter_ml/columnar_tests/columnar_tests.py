from sklearn import metrics
import numpy as np
import time
from scipy import stats
from mlxtend.evaluate import permutation_test

class DataSanitization(): 
    def __init__(self, data):
        """
        Initialize data.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        self.data = data
        
    def is_complete(self, column):
        """
        Return true if the column is complete.

        Args:
            self: (todo): write your description
            column: (int): write your description
        """
        return self.data[column].isnull().sum() == 0

    def has_completeness(self, column, threshold):
        """
        Returns true if the given column has the given threshold.

        Args:
            self: (todo): write your description
            column: (str): write your description
            threshold: (float): write your description
        """
        return self.data[column].isnull().sum()/len(self.data) > threshold

    def is_unique(self, column):
        """
        Returns true if column is unique.

        Args:
            self: (todo): write your description
            column: (array): write your description
        """
        return len(self.data[column].unique())/len(self.data) == 1

    def has_uniqueness(self, column, threshold):
        """
        Returns true if the given column has the given threshold.

        Args:
            self: (todo): write your description
            column: (array): write your description
            threshold: (float): write your description
        """
        return len(self.data[column].unique())/len(self.data) > threshold

    def is_in_range(self, column, lower_bound, upper_bound, threshold):
        """
        Returns true if the given range is in the range.

        Args:
            self: (todo): write your description
            column: (str): write your description
            lower_bound: (int): write your description
            upper_bound: (todo): write your description
            threshold: (float): write your description
        """
        return self.data[(self.data[column] <= upper_bound) & (self.data[column] >= lower_bound)]/len(self.data) > threshold

    def is_non_negative(self, column):
        """
        Returns true if column is negative false otherwise.

        Args:
            self: (todo): write your description
            column: (str): write your description
        """
        return self.data[self.data[column] > 0]

    def is_less_than(self, column_one, column_two):
        """
        Returns true if all columns in the same

        Args:
            self: (todo): write your description
            column_one: (str): write your description
            column_two: (str): write your description
        """
        return self.data[self.data[column_one] < self.data[column_two]].all()

class ColumnarData():
    def __init__(self, historical_data, new_data):
        """
        Initialize historical data.

        Args:
            self: (todo): write your description
            historical_data: (todo): write your description
            new_data: (todo): write your description
        """
        self.new_data = new_data
        self.historical_data = historical_data

    def mean_similarity(self, column, tolerance=2):
        """
        Computes the mean of the data.

        Args:
            self: (todo): write your description
            column: (array): write your description
            tolerance: (float): write your description
        """
        new_mean = float(np.mean(self.new_data[column]))
        old_mean = float(np.mean(self.historical_data[column]))
        std = float(np.std(self.historical_data[column]))
        upper_bound = old_mean + (std * tolerance)
        lower_bound = old_mean - (std * tolerance)
        if new_mean < lower_bound:
            return False
        elif new_mean > upper_bound:
            return False
        else:
            return True

    def median_similarity(self, column, tolerance=2):
        """
        Computes the median of a column.

        Args:
            self: (todo): write your description
            column: (array): write your description
            tolerance: (float): write your description
        """
        new_median = float(np.median(self.new_data[column]))
        old_median = float(np.median(self.historical_data[column]))
        iqr = float(stats.iqr(self.historical_data[column]))
        upper_bound = old_median + (iqr * tolerance)
        lower_bound = old_median - (iqr * tolerance)
        if new_median < lower_bound:
            return False
        elif new_median > upper_bound:
            return False
        else:
            return True

    def trimean(self, data):
        """
        Calculate the median of data

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        q1 = float(np.quantile(data, 0.25))
        q3 = float(np.quantile(data, 0.75))
        median = float(np.median(data))
        return (q1 + 2*median + q3)/4

    def trimean_absolute_deviation(self, data):
        """
        Return the deviation of the data.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        trimean = self.trimean(data)
        numerator = [abs(elem - trimean) for elem in data]
        return sum(numerator)/len(data)

    def trimean_similarity(self, column, tolerance=2):
        """
        Trime similarity.

        Args:
            self: (todo): write your description
            column: (array): write your description
            tolerance: (float): write your description
        """
        new_trimean = self.trimean(self.new_data[column])
        old_trimean = self.trimean(self.historical_data[column])
        tad = self.trimean_absolute_deviation(self.historical_data[column])
        upper_bound = old_trimean + (tad * tolerance)
        lower_bound = old_trimean - (tad * tolerance)
        if new_trimean < lower_bound:
            return False
        if new_trimean > upper_bound:
            return False
        else:
            return True
    
    def is_normal(self, column):
        """
        Return true if column is a column

        Args:
            self: (todo): write your description
            column: (str): write your description
        """
        new_data_result = stats.normaltest(self.new_data[column])
        historical_data_result = stats.normaltest(self.historical_data[column])
        if new_data_result.pvalue > 0.05 and historical_data_result.pvalue > 0.05:
            return True
        return False
    
    def pearson_similar_correlation(self, column,
                                     correlation_lower_bound,
                                     pvalue_threshold=0.05,
                                     num_rounds=3):
        """
        Calculate the correlation coefficient.

        Args:
            self: (todo): write your description
            column: (str): write your description
            correlation_lower_bound: (todo): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        correlation_info = stats.pearsonr(self.new_data[column],
                                          self.historical_data[column])
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.pearsonr(x, y)[0],
            seed=0)
        if p_value > pvalue_threshold:
            return False
        if correlation_info[0] < correlation_lower_bound:
            return False
        return True

    def spearman_similar_correlation(self, column,
                                      correlation_lower_bound,
                                      pvalue_threshold=0.05,
                                      num_rounds=3):
        """
        Calculate correlation coefficient.

        Args:
            self: (todo): write your description
            column: (str): write your description
            correlation_lower_bound: (todo): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        correlation_info = stats.spearmanr(self.new_data[column],
                                           self.historical_data[column])
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.spearmanr(x, y).correlation,
            seed=0)
        if p_value > pvalue_threshold:
            return False
        if correlation_info.correlation < correlation_lower_bound:
            return False
        return True

    def wilcoxon_similar_distribution(self, column,
                                       pvalue_threshold=0.05,
                                       num_rounds=3):
        """
        Wilcoxon similarity.

        Args:
            self: (todo): write your description
            column: (str): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.wilcoxon(x, y).statistic,
            seed=0)
        if p_value < pvalue_threshold:
            return False
        return True
        
    def ks_2samp_similar_distribution(self, column,
                                       pvalue_threshold=0.05,
                                       num_rounds=3):
        """
        Compute the similarity between two columns.

        Args:
            self: (todo): write your description
            column: (str): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.ks_2samp(x, y).statistic,
            seed=0)
        if p_value < pvalue_threshold:
            return False
        return True

    def kruskal_similar_distribution(self, column,
                                      pvalue_threshold=0.05,
                                      num_rounds=3):
        """
        Compute kruskal score.

        Args:
            self: (todo): write your description
            column: (str): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.kruskal(x, y).statistic,
            seed=0)
        if p_value < pvalue_threshold:
            return False
        return True

    def mann_whitney_u_similar_distribution(self, column,
                                            pvalue_threshold=0.05,
                                            num_rounds=3):
        """
        Determine whether a column is a 2d histogram.

        Args:
            self: (todo): write your description
            column: (str): write your description
            pvalue_threshold: (float): write your description
            num_rounds: (int): write your description
        """
        p_value = permutation_test(
            self.new_data[column],
            self.historical_data[column],
            method="approximate",
            num_rounds=num_rounds,
            func=lambda x, y: stats.mannwhitneyu(x, y).statistic,
            seed=0)

        if p_value < pvalue_threshold:
            return False
        return True
