import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_val_score

class ModelClassificationTestSuite():
    def __init__(self, clf_name, clf_metadata, data_filename):
        clf, metadata, colum_names, target_name, test_data = self.get_parameters(
            clf_name, clf_metadata, data_filename)
        self.clf = clf
        self.data_filename
        self.metadata = metadata
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)

    # potentially include hyper parameters from the model
    # algorithm could be stored in metadata
    def get_parameters(self, clf_name, clf_metadata, data_filename):
        clf = joblib.load(clf_name)
        metadata = json.load(open(clf_metadata, "r"))
        column_names = metadata["column_names"]
        target_name = metadata["target_name"]
        test_data = pd.read_csv(data_name)
        return clf, metadata, column_names, target_name, test_data

    def precision_lower_boundary_per_class(self, lower_boundary):
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.precision_score(y_class, y_pred_class) < class_info["precision_score"]:
                return False
        return True

    def recall_lower_boundary_per_class(self, lower_boundary):
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.recall_score(y_class, y_pred_class) < class_info["recall_score"]:
                return False
        return True

    def f1_lower_boundary_per_class(self, clf, test_data, target_name, column_names, lower_boundary):
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.f1_score(y_class, y_pred_class) < class_info["f1_score"]:
                return False
        return True

    def classifier_testing(self, precision_lower_boundary, recall_lower_boundary, f1_lower_boundary):
        precision_test = self.precision_lower_boundary_per_class(precision_lower_boundary)
        recall_test = self.recall_lower_boundary_per_class(recall_lower_boundary)
        f1_test = self.f1_lower_boundary_per_class(f1_lower_boundary)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def run_time_stress_test(self, performance_boundary):
        for performance_info in performance_boundary:
            n = int(performance_info["sample_size"])
            max_run_time = float(performance_info["max_run_time"])
            data = self.X.sample(n, replace=True)
            start_time = time.time()
            self.clf.predict(data)
            model_run_time = time.time() - start_time
            if model_run_time > run_time:
                return False
        return True

class ClassifierComparison():
    def __init__(self, clf_one_name, clf_one_metadata, clf_two_name, clf_two_metadata, data_filename):
        clf_one, metadata_one, colum_names, target_name, test_data = self.get_parameters(
            clf_one_name, clf_one_metadata, data_filename)
        clf_two, metadata_two, colum_names, target_name, test_data = self.get_parameters(
            clf_two_name, clf_two_metadata, data_filename)
        self.clf_one = clf_one
        self.clf_two = clf_two
        self.data_filename
        self.metadata_one = metadata_one
        self.metadata_two = metadata_two
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)
        
    def two_model_prediction_run_time_stress_test(self, performance_boundary):
        for performance_info in performance_boundary:
            n = int(performance_info["sample_size"])
            data = self.X.sample(n, replace=True)
            start_time = time.time()
            self.clf_one.predict(data)
            model_one_run_time = time.time() - start_time
            start_time = time.time()
            self.clf_two.predict(data)
            model_two_run_time = time.time() - start_time
            # we assume model one should be faster than model two
            if model_one_run_time > model_two_run_time:
                return False
        return True

    def precision_per_class(self, clf, test_data, target_name, column_names):
        y = test_data[target_name]
        classes = set(y)
        y_pred = clf.predict(test_data[column_names])
        precision = {}
        for klass in classes:
            y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
            y_class = y[y == klass]
            precision[klass] = metrics.precision_score(y_class, y_pred_class) 
        return precision

    def recall_per_class(self, clf, test_data, target_name, column_names):
        y = test_data[target_name]
        classes = set(y)
        y_pred = clf.predict(test_data[column_names])
        recall = {}
        for klass in classes:
            y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
            y_class = y[y == klass]
            recall[klass] = metrics.recall_score(y_class, y_pred_class)
        return recall

    def f1_per_class(self, clf, test_data, target_name, column_names):
        y = test_data[target_name]
        classes = set(y)
        y_pred = clf.predict(test_data[column_names])
        f1 = {}
        for klass in classes:
            y_pred_class = np.take(y_pred, y[y == klass].index, axis=0)
            y_class = y[y == klass]
            f1[klass] = metrics.f1_score(y_class, y_pred_class)
        return f1

    def two_model_classifier_testing(self):
        precision_one_test = self.precision_per_class(self.clf_one)
        recall_one_test = self.recall_per_class(self.clf_one)
        f1_one_test = self.f1_per_class(self.clf_one)
        precision_two_test = precision_per_class(self.clf_two)
        recall_two_test = recall_per_class(self.clf_two)
        f1_two_test = f1_per_class(self.clf_two)

        precision_result =  precision_one_test > precision_two_test
        recall_result = recall_one_test > recall_two_test
        f1_result = f1_one_test > f1_two_test
        if precision_result and recall_result and f1_result:
            return True
        else:
            return False
