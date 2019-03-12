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
        
    def similiar_correlation(self, column, correlation_lower_bound, pvalue_threshold=0.05):
        correlation_info = stats.spearmanr(self.new_data[column_name], self.historical_data[column_name])
        if correlation_info.pvalue > pvalue_threshold:
            return False
        if correlation_info.correlation < correlation_lower_bound:
            return False
        return True

    def similiar_distribution(self, column, pvalue_threshold=0.05):
        distribution_info = stats.ks_2samp(self.new_data[column_name],
                                           self.historical_data[column_name])
        if distribution_info.pvalue < pvalue_threshold:
            return False
        return True
