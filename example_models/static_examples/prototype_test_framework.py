import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_val_score

# classification tests
class ModelClassificationTestSuite():
    def __init__(self, clf_name, clf_metadata, data_filename):
        """
        Initialize the class with the target metadata.

        Args:
            self: (todo): write your description
            clf_name: (str): write your description
            clf_metadata: (todo): write your description
            data_filename: (str): write your description
        """
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
        """
        Get the metadata file. csv file.

        Args:
            self: (todo): write your description
            clf_name: (str): write your description
            clf_metadata: (todo): write your description
            data_filename: (str): write your description
        """
        clf = joblib.load(clf_name)
        metadata = json.load(open(clf_metadata, "r"))
        column_names = metadata["column_names"]
        target_name = metadata["target_name"]
        test_data = pd.read_csv(data_name)
        return clf, metadata, column_names, target_name, test_data

    def precision_lower_boundary_per_class(self, lower_boundary):
        """
        Precision class labels

        Args:
            self: (todo): write your description
            lower_boundary: (todo): write your description
        """
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.precision_score(y_class, y_pred_class) < class_info["precision_score"]:
                return False
        return True

    def recall_lower_boundary_per_class(self, lower_boundary):
        """
        Recall_boundary class labels

        Args:
            self: (todo): write your description
            lower_boundary: (todo): write your description
        """
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.recall_score(y_class, y_pred_class) < class_info["recall_score"]:
                return False
        return True

    def f1_lower_boundary_per_class(self, clf, test_data, target_name, column_names, lower_boundary):
        """
        R compute the f1 score.

        Args:
            self: (todo): write your description
            clf: (todo): write your description
            test_data: (todo): write your description
            target_name: (str): write your description
            column_names: (list): write your description
            lower_boundary: (todo): write your description
        """
        y_pred = self.clf.predict(self.X)
        for class_info in lower_boundary["per_class"]:
            klass = class_info["class"]
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metrics.f1_score(y_class, y_pred_class) < class_info["f1_score"]:
                return False
        return True

    def classifier_testing(self, precision_lower_boundary, recall_lower_boundary, f1_lower_boundary):
        """
        Returns true if the precision_bound_lower_bound_boundary.

        Args:
            self: (todo): write your description
            precision_lower_boundary: (bool): write your description
            recall_lower_boundary: (bool): write your description
            f1_lower_boundary: (todo): write your description
        """
        precision_test = self.precision_lower_boundary_per_class(precision_lower_boundary)
        recall_test = self.recall_lower_boundary_per_class(recall_lower_boundary)
        f1_test = self.f1_lower_boundary_per_class(f1_lower_boundary)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def run_time_stress_test(self, performance_boundary):
        """
        Runs all the given performance command.

        Args:
            self: (todo): write your description
            performance_boundary: (todo): write your description
        """
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

# post training - 
# todo: add model metric outside of some standard deviation
# for many models
# is the model non-empty
# is the model deserializable

# test against training and scoring

class ModelRegressionTestSuite():
    def __init__(self, reg_name, reg_metadata, data_filename):
        """
        Initialize the target.

        Args:
            self: (todo): write your description
            reg_name: (str): write your description
            reg_metadata: (todo): write your description
            data_filename: (str): write your description
        """
        reg, reg_metadata, colum_names, target_name, test_data = self.get_parameters(
            reg_name, reg_metadata, data_filename)
        self.reg = reg
        self.data_filename
        self.metadata = metadata
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        
    def get_parameters(self, reg_name, reg_metadata, data_filename):
        """
        Get metadata from a dictionary.

        Args:
            self: (todo): write your description
            reg_name: (str): write your description
            reg_metadata: (str): write your description
            data_filename: (str): write your description
        """
        reg = joblib.load(reg_name)
        metadata = json.load(open(reg_metadata, "r"))
        column_names = metadata["column_names"]
        target_name = metadata["target_name"]
        test_data = pd.read_csv(data_name)
        return reg, metadata, column_names, target_name, test_data

    def mse_upper_boundary(upper_boundary):
        """
        Evaluate the squared squared boundary.

        Args:
            upper_boundary: (todo): write your description
        """
        y_pred = self.reg.predict(self.X)
        if metrics.mean_squared_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def mae_upper_boundary(upper_boundary):
        """
        R compute the upper bound on the upper and lower bound.

        Args:
            upper_boundary: (todo): write your description
        """
        y_pred = self.reg.predict(self.X)
        if metrics.median_absolute_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def regression_testing(mse_upper_boundary, mae_upper_boundary):
        """
        Determine whether the upper and upper and upper bound on the upper and false otherwise.

        Args:
            mse_upper_boundary: (todo): write your description
            mae_upper_boundary: (int): write your description
        """
        mse_test = self.mse_upper_boundary(mse_upper_boundary)
        mae_test = self.mae_upper_boundary(mae_upper_boundary)
        if mse_test and mae_test:
            return True
        else:
            return False

    def run_time_stress_test(self, performance_boundary):
        """
        Runs the given performance of the given performance.

        Args:
            self: (todo): write your description
            performance_boundary: (todo): write your description
        """
        for performance_info in performance_boundary:
            n = int(performance_info["sample_size"])
            max_run_time = float(performance_info["max_run_time"])
            data = self.X.sample(n, replace=True)
            start_time = time.time()
            self.reg.predict(data)
            model_run_time = time.time() - start_time
            if model_run_time > run_time:
                return False
        return True

class ClassifierComparison():
    def __init__(self, clf_one_name, clf_one_metadata, clf_two_name, clf_two_metadata, data_filename):
        """
        Initialize a single class

        Args:
            self: (todo): write your description
            clf_one_name: (str): write your description
            clf_one_metadata: (todo): write your description
            clf_two_name: (str): write your description
            clf_two_metadata: (todo): write your description
            data_filename: (str): write your description
        """
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
        """
        Predict whether a prediction is the prediction.

        Args:
            self: (todo): write your description
            performance_boundary: (int): write your description
        """
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
        """
        Precision classifier.

        Args:
            self: (todo): write your description
            clf: (todo): write your description
            test_data: (todo): write your description
            target_name: (str): write your description
            column_names: (str): write your description
        """
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
        """
        Calculate the recall.

        Args:
            self: (todo): write your description
            clf: (todo): write your description
            test_data: (todo): write your description
            target_name: (str): write your description
            column_names: (str): write your description
        """
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
        """
        Compute the f1 test.

        Args:
            self: (todo): write your description
            clf: (todo): write your description
            test_data: (todo): write your description
            target_name: (str): write your description
            column_names: (str): write your description
        """
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
        """
        Returns true if precision classifier is_classifier false otherwise.

        Args:
            self: (todo): write your description
        """
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

class RegressionComparison():
    def __init__(self, reg_one_name, reg_one_metadata, reg_two_name, reg_two_metadata, data_filename):
        """
        Initialize the target.

        Args:
            self: (todo): write your description
            reg_one_name: (str): write your description
            reg_one_metadata: (todo): write your description
            reg_two_name: (str): write your description
            reg_two_metadata: (todo): write your description
            data_filename: (str): write your description
        """
        reg_one, metadata_one, colum_names, target_name, test_data = self.get_parameters(
            reg_one_name, reg_one_metadata, data_filename)
        reg_two, metadata_two, colum_names, target_name, test_data = self.get_parameters(
            reg_two_name, reg_two_metadata, data_filename)
        self.reg_one = reg_one
        self.reg_two = reg_two
        self.data_filename
        self.metadata_one = metadata_one
        self.metadata_two = metadata_two
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        
    def two_model_prediction_run_time_stress_test(self, performance_boundary):
        """
        Returns true if a prediction is a prediction.

        Args:
            self: (todo): write your description
            performance_boundary: (int): write your description
        """
        for performance_info in performance_boundary:
            n = int(performance_info["sample_size"])
            data = self.X.sample(n, replace=True)
            start_time = time.time()
            self.reg_one.predict(data)
            model_one_run_time = time.time() - start_time
            start_time = time.time()
            self.reg_two.predict(data)
            model_two_run_time = time.time() - start_time
            # we assume model one should be faster than model two
            if model_one_run_time > model_two_run_time:
                return False
        return True

    def mse_result(self, reg):
        """
        Compute the squared error.

        Args:
            self: (todo): write your description
            reg: (todo): write your description
        """
        y_pred = reg.predict(self.X)
        return metrics.mean_squared_error(self.y, y_pred)

    def mae_result(self, reg):
        """
        Mae_result

        Args:
            self: (todo): write your description
            reg: (todo): write your description
        """
        y_pred = reg.predict(self.X)
        return metrics.median_absolute_error(self.y, y_pred)

    def two_model_regression_testing(self):
        """
        Determine if two regression are equal.

        Args:
            self: (todo): write your description
        """
        mse_one_test = self.mse_result(self.reg_one)
        mae_one_test = self.mae_result(self.reg_one)
        mse_two_test = self.mse_result(self.reg_two)
        mae_two_test = self.mae_result(self.reg_two)
        if mse_one_test < mse_two_test and mae_one_test < mae_two_test:
            return True
        else:
            return False

# data tests
class DataSanitization(): 
    def __init__(self, data_filename):
        """
        Initialize data_filename.

        Args:
            self: (todo): write your description
            data_filename: (str): write your description
        """
        self.data_filename
        self.data = pd.read_csv(data_filename)
        
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

    def has_uniqueness(column, threshold):
        """
        Returns true if the given threshold.

        Args:
            column: (array): write your description
            threshold: (float): write your description
        """
        return len(self.data[column].unique())/len(self.data) > threshold

    def is_in_range(column, lower_bound, upper_bound, threshold):
        """
        Returns true if the given range is within the range.

        Args:
            column: (str): write your description
            lower_bound: (int): write your description
            upper_bound: (todo): write your description
            threshold: (float): write your description
        """
        return self.data[(self.data[column] <= upper_bound) & (self.data[column] >= lower_bound)]/len(self.data) > threshold

    def is_non_negative(column):
        """
        Returns true if column is negative false otherwise.

        Args:
            column: (str): write your description
        """
        return self.data[self.data[column] > 0]

    def is_less_than(column_one, column_two):
        """
        Returns true if all columns in the same

        Args:
            column_one: (str): write your description
            column_two: (str): write your description
        """
        return self.data[self.data[column_one] < self.data[column_two]].all()

# memoryful tests
class StructuralData():
    def __init__(self, metadata, data_filename):
        """
        Initialize the metadata file.

        Args:
            self: (todo): write your description
            metadata: (dict): write your description
            data_filename: (str): write your description
        """
        metadata, column_names, target_name, test_data = self.get_parameters(
            metadata, data_filename)
        self.data_filename
        self.metadata = metadata
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]

    def get_parameters(self, metadata, data_filename):
        """
        Get metadata from metadata file.

        Args:
            self: (todo): write your description
            metadata: (todo): write your description
            data_filename: (str): write your description
        """
        metadata = json.load(open(clf_metadata, "r"))
        column_names = metadata["column_names"]
        target_name = metadata["target_name"]
        test_data = pd.read_csv(data_name)
        return metadata, column_names, target_name, test_data

    def reg_clustering(self, data, columns, target):
        """
        Compute the k - means clustering

        Args:
            self: (todo): write your description
            data: (array): write your description
            columns: (list): write your description
            target: (str): write your description
        """
        k_measures = []
        for k in range(2, 12):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.X, self.y)
            y_pred = knn.predict(self.X)
            k_measures.append((k, metrics.mean_squared_error(self.y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def reg_similar_clustering(self, absolute_distance, new_data, historical_data, column_names, target_name):
        """
        Computes the k - correlations.

        Args:
            self: (todo): write your description
            absolute_distance: (todo): write your description
            new_data: (todo): write your description
            historical_data: (todo): write your description
            column_names: (str): write your description
            target_name: (str): write your description
        """
        historical_k = reg_clustering(historical_data, column_names, target_name)
        new_k = reg_clustering(new_data, column_names, target_name)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True

    # this was never updated
    def cls_clustering(self):
        """
        Clust clustering

        Args:
            self: (todo): write your description
        """
        k_measures = []
        for k in range(2, 12):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.X, self.y)
            y_pred = knn.predict(self.X)
            k_measures.append((k, metrics.mean_squared_error(self.y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def cls_similiar_clustering(absolute_distance, new_data, historical_data, column_names, target_name):
        """
        Computes the k - means clustering.

        Args:
            absolute_distance: (todo): write your description
            new_data: (todo): write your description
            historical_data: (todo): write your description
            column_names: (str): write your description
            target_name: (str): write your description
        """
        historical_k = cls_clustering(historical_data, column_names, target_name)
        new_k = cls_clustering(new_data, column_names, target_name)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True

# this needs work
class ColumnarData():
    def similiar_correlation(correlation_lower_bound, new_data, historical_data, column_names, pvalue_threshold=0.05):
        """
        Simulate the correlation. correlation.

        Args:
            correlation_lower_bound: (todo): write your description
            new_data: (todo): write your description
            historical_data: (str): write your description
            column_names: (list): write your description
            pvalue_threshold: (float): write your description
        """
        for column_name in column_names:
            correlation_info = stats.spearmanr(new_data[column_name], historical_data[column_name])
            if correlation_info.pvalue > pvalue_threshold:
                return False
            if correlation_info.correlation < correlation_lower_bound:
                return False
        return True

    def similiar_distribution(new_data, historical_data, column_names, pvalue_threshold=0.05):
        """
        Simulate the distribution of the data.

        Args:
            new_data: (todo): write your description
            historical_data: (bool): write your description
            column_names: (list): write your description
            pvalue_threshold: (float): write your description
        """
        for column_name in column_names:
            distribution_info = stats.ks_2samp(new_data[column_name], historical_data[column_name])
            if correlation_info.pvalue < pvalue_threshold:
                return False
        return True

# does the preprocessing break?
# does the model build?
# does the model meet some threshold?
# add memoryful tests for measures over time (like over several days)
