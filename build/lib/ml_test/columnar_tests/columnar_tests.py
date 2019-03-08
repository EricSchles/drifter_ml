import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_val_score

class DataSanitization(): 
    def __init__(self, data_filename):
        self.data_filename
        self.data = pd.read_csv(data_filename)
        
    def is_complete(self, column):
        return self.data[column].isnull().sum() == 0

    def has_completeness(self, column, threshold):
        return self.data[column].isnull().sum()/len(self.data) > threshold

    def is_unique(self, column):
        return len(self.data[column].unique())/len(self.data) == 1

    def has_uniqueness(column, threshold):
        return len(self.data[column].unique())/len(self.data) > threshold

    def is_in_range(column, lower_bound, upper_bound, threshold):
        return self.data[(self.data[column] <= upper_bound) & (self.data[column] >= lower_bound)]/len(self.data) > threshold

    def is_non_negative(column):
        return self.data[self.data[column] > 0]

    def is_less_than(column_one, column_two):
        return self.data[self.data[column_one] < self.data[column_two]].all()

class ColumnarData():
    def similiar_correlation(correlation_lower_bound, new_data, historical_data, column_names, pvalue_threshold=0.05):
        for column_name in column_names:
            correlation_info = stats.spearmanr(new_data[column_name], historical_data[column_name])
            if correlation_info.pvalue > pvalue_threshold:
                return False
            if correlation_info.correlation < correlation_lower_bound:
                return False
        return True

    def similiar_distribution(new_data, historical_data, column_names, pvalue_threshold=0.05):
        for column_name in column_names:
            distribution_info = stats.ks_2samp(new_data[column_name], historical_data[column_name])
            if correlation_info.pvalue < pvalue_threshold:
                return False
        return True
