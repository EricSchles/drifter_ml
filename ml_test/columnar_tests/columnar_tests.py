import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats

class DataSanitization(): 
    def __init__(self, data):
        self.data = data
        
    def is_complete(self, column):
        return self.data[column].isnull().sum() == 0

    def has_completeness(self, column, threshold):
        return self.data[column].isnull().sum()/len(self.data) > threshold

    def is_unique(self, column):
        return len(self.data[column].unique())/len(self.data) == 1

    def has_uniqueness(self, column, threshold):
        return len(self.data[column].unique())/len(self.data) > threshold

    def is_in_range(self, column, lower_bound, upper_bound, threshold):
        return self.data[(self.data[column] <= upper_bound) & (self.data[column] >= lower_bound)]/len(self.data) > threshold

    def is_non_negative(self, column):
        return self.data[self.data[column] > 0]

    def is_less_than(self, column_one, column_two):
        return self.data[self.data[column_one] < self.data[column_two]].all()

class ColumnarData():
    def __init__(self, historical_data, new_data):
        self.new_data = new_data
        self.historical_data = historical_data

    def is_normal(self, column):
        new_data_result = stats.normaltest(self.new_data[column])
        historical_data_result = stats.normaltest(self.historical_data[column])
        if new_data_result.pvalue > 0.05 and historical_data_result.pvalue > 0.05:
            return True
        return False
    
    def pearson_similiar_correlation(self, column, correlation_lower_bound, pvalue_threshold=0.05):
        if not self.is_normal(column):
            raise Exception("""
            Data is likely not normally distributed and therefore pearson is not
            a valid test to run""")
        correlation_info = stats.pearsonr(self.new_data[column],
                                          self.historical_data[column])
        if correlation_info.pvalue > pvalue_threshold:
            return False
        if correlation_info.correlation < correlation_lower_bound:
            return False
        return True

    def spearman_similiar_correlation(self, column, correlation_lower_bound, pvalue_threshold=0.05):
        correlation_info = stats.spearmanr(self.new_data[column],
                                           self.historical_data[column])
        if correlation_info.pvalue > pvalue_threshold:
            return False
        if correlation_info.correlation < correlation_lower_bound:
            return False
        return True

    def ks_2samp_similiar_distribution(self, column, pvalue_threshold=0.05):
        distribution_info = stats.ks_2samp(self.new_data[column],
                                           self.historical_data[column])
        if distribution_info.pvalue < pvalue_threshold:
            return False
        return True
