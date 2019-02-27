import joblib
import json
from sklearn import metrics
import numpy as np
import time

def generate_test_data():
    pass

def precision_lower_boundary_per_class(clf, test_data, target_names, column_names, lower_boundary):
    y = test_data[target_names]
    y_pred = clf.predict(test_data[column_names])
    
    for class_info in lower_boundary["per_class"]:
        klass = class_info["class"]
        y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
        y_class = y[y == klass]
        if metrics.precision_score(y_class, y_pred_class) < class_info["precision_score"]:
            return False
    return True

def recall_lower_boundary_per_class(clf, test_data, target_names, column_names, lower_boundary):
    y = test_data[target_names]
    y_pred = clf.predict(test_data[column_names])
    
    for class_info in lower_boundary["per_class"]:
        klass = class_info["class"]
        y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
        y_class = y[y == klass]
        if metrics.recall_score(y_class, y_pred_class) < class_info["recall_score"]:
            return False
    return True

def f1_lower_boundary_per_class(clf, test_data, target_names, column_names, lower_boundary):
    y = test_data[target_names]
    y_pred = clf.predict(test_data[column_names])
    
    for class_info in lower_boundary["per_class"]:
        klass = class_info["class"]
        y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
        y_class = y[y == klass]
        if metrics.f1_score(y_class, y_pred_class) < class_info["f1_score"]:
            return False
    return True

def mse_upper_boundary(reg, test_data, target_names, column_names, upper_boundary):
    y = test_data[target_names]
    y_pred = reg.predict(test_data[column_names])
    if metrics.mean_squared_error(y, y_pred) > upper_boundary:
        return False
    return True

def mae_upper_boundary(reg, test_data, target_names, column_names, upper_boundary):
    y = test_data[target_names]
    y_pred = reg.predict(test_data[column_names])
    if metrics.median_absolute_error(y, y_pred) > upper_boundary:
        return False
    return True

def prediction_run_time_stress_test(model, test_data, column_names, performance_boundary):
    X = test_data[column_names]
    for performance_info in performance_boundary:
        n = int(performance_info["sample_size"])
        max_run_time = float(performance_info["max_run_time"])
        data = X.sample(n, replace=True)
        start_time = time.time()
        model.predict(data)
        model_run_time = time.time() - start_time
        if model_run_time > run_time:
            return False
    return True

def is_complete(data, column):
    return data[column].isnull().sum() == 0

def has_completeness(data, column, threshold):
    return data[column].isnull().sum()/len(data) > threshold

def is_unique(data, column):
    return len(data[column].unique())/len(df) == 1

def has_uniqueness(data, column, threshold):
    return len(data[column].unique())/len(df) > threshold

def is_in_range(data, column, lower_bound, upper_bound, threshold):
    return data[(data[column] <= upper_bound) & (data[column] >= lower_bound)]/len(data) > threshold

def is_non_negative(data, column):
    return data[data[column] > 0]

def is_less_than(data, column_one, column_two):
    return data[data[column_one] < data[column_two]].all()


