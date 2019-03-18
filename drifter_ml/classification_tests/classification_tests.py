import joblib
import json
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn import base
from typing import Optional
from functools import partial

class FixedClassificationMetrics():
    def __init__(self):
        pass
    
    def precision_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.precision_score(y_true,
                                           y_pred,
                                           labels=labels,
                                           pos_label=pos_label,
                                           average=average,
                                           sample_weight=sample_weight)

    def recall_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.recall_score(y_true,
                                        y_pred,
                                        labels=labels,
                                        pos_label=pos_label,
                                        average=average,
                                        sample_weight=sample_weight)

    def f1_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.f1_score(y_true,
                                    y_pred,
                                    labels=labels,
                                    pos_label=pos_label,
                                    average=average,
                                    sample_weight=sample_weight)

class ClassificationTests(FixedClassificationMetrics):
    def __init__(self,
                 clf,
                 test_data,
                 target_name,
                 column_names):
        self.clf = clf
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)

    def precision_cv(self):
        precision_score = self.precision_score()
        precision = metrics.make_scorer(precision_score)
        return cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(precision))


    def cross_val_precision_anomaly_detection(self, tolerance, cv=3):
        scores = self.precision_cv()
        avg = st.mean(scores)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviances_from_avg:
            if deviance > tolerance:
                return False
        return True

    def cross_val_precision_lower_boundary(self, lower_boundary, cv=3):
        scores = self.precision_cv()
        for score in scores:
            if score < lower_boundary:
                return False
        return True
    
    def recall_cv(self):
        recall_score = self.recall_score()
        recall = metrics.make_scorer(recall_score)
        return cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(recall))
    
    def cross_val_recall_anomaly_detection(self, tolerance, cv=3):
        scores = self.recall_cv()
        avg = st.mean(scores)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviances_from_avg:
            if deviance > tolerance:
                return False
        return True

    def cross_val_recall_lower_boundary(self, lower_boundary, cv=3):
        scores = self.recall_cv()
        for score in scores:
            if score < lower_boundary:
                return False
        return True


    def f1_cv(self):
        f1_score = self.f1_score()
        f1 = metrics.make_scorer(f1_score)
        return cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(f1))
    
    def cross_val_f1_anomaly_detection(self, tolerance, cv=3):
        scores = self.f1_cv()
        avg = st.mean(scores)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviances_from_avg:
            if deviance > tolerance:
                return False
        return True

    def cross_val_f1_lower_boundary(self, lower_boundary, cv=3):
        scores = self.f1_cv()
        for score in scores:
            if score < lower_boundary:
                return False
        return True

    def cross_val_classifier_testing(self,
                                     precision_lower_boundary: float,
                                     recall_lower_boundary: float,
                                     f1_lower_boundary: float,
                                     cv=3):
        precision_test = self.cross_val_precision_lower_boundary(
            precision_lower_boundary, cv=cv)
        recall_test = self.cross_val_recall_lower_boundary(
            recall_lower_boundary, cv=cv)
        f1_test = self.cross_val_f1_lower_boundary(
            f1_lower_boundary, cv=cv)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def describe_scores(self, scores, method):
        if method == "normal":
            return np.mean(scores), np.std(scores)
        elif method == "ranked":
            return np.median(scores), stats.iqr(scores)
        
    def auto_cross_val_precision_anomaly_detection(self, tolerance, method="normal", cv=10):
        scores = self.precision_cv()
        center, spread = self.describe_scores(scores, method)
        for score in scores:
            if score < center-(spread*tolerance):
                return False
        return True

    def auto_cross_val_f1_anomaly_detection(self, tolerance, method="normal", cv=10):
        scores = self.f1_cv()
        center, spread = self.describe_scores(scores, method)
        for score in scores:
            if score < center-(spread*tolerance):
                return False
        return True

    def auto_cross_val_recall_anomaly_detection(self, tolerance, method="normal", cv=3):
        scores = self.recall_cv()
        center, spread = self.describe_scores(scores, method)
        for score in scores:
            if score < center-(spread*tolerance):
                return False
        return True

    def auto_cross_val_classifier_testing(self,
                                          precision_lower_boundary: int,
                                          recall_lower_boundary: int,
                                          f1_lower_boundary: int,
                                          cv=10):
        precision_test = self.auto_cross_val_precision_lower_boundary(
            precision_lower_boundary, cv=cv)
        recall_test = self.auto_cross_val_recall_lower_boundary(
            recall_lower_boundary, cv=cv)
        f1_test = self.auto_cross_val_f1_lower_boundary(
            f1_lower_boundary, cv=cv)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    # potentially include hyper parameters from the model
    # algorithm could be stored in metadata
    def precision_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if self.precision_score(y_class, y_pred_class) < lower_boundary[klass]:
                return False
        return True

    def recall_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if self.recall_score(y_class, y_pred_class) < lower_boundary[klass]:
                return False
        return True

    def f1_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if self.f1_score(y_class, y_pred_class) < lower_boundary[klass]:
                return False
        return True
        
    def classifier_testing(self,
                           precision_lower_boundary: dict,
                           recall_lower_boundary: dict,
                           f1_lower_boundary: dict):
        precision_test = self.precision_lower_boundary_per_class(precision_lower_boundary)
        recall_test = self.recall_lower_boundary_per_class(recall_lower_boundary)
        f1_test = self.f1_lower_boundary_per_class(f1_lower_boundary)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def run_time_stress_test(self, performance_boundary: dict):
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

class ClassifierComparison(FixedClassificationMetrics):
    def __init__(self,
                 clf_one,
                 clf_two,
                 test_data,
                 target_name,
                 column_names):
        self.clf_one = clf_one
        self.clf_two = clf_two
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
    
    def precision_per_class(self, clf):
        precision_score = self.precision_score()
        precision_scorer = metrics.make_scorer(precision_score)
        y_pred = clf.predict(self.X)
        precision = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            precision[klass] = precision_scorer(y_class, y_pred_class) 
        return precision

    def recall_per_class(self, clf):
        recall_score = self.recall_score()
        recall_scorer = metrics.make_scorer(recall_score)
        y_pred = clf.predict(self.X)
        recall = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            recall[klass] = recall_scorer(y_class, y_pred_class)
        return recall

    def f1_per_class(self, clf):
        f1_score = self.f1_score()
        f1_scorer = metrics.make_scorer(f1_score)
        y_pred = clf.predict(self.X)
        f1 = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            f1[klass] = f1_scorer(y_class, y_pred_class)
        return f1

    def two_model_classifier_testing(self):
        precision_one_test = self.precision_per_class(self.clf_one)
        recall_one_test = self.recall_per_class(self.clf_one)
        f1_one_test = self.f1_per_class(self.clf_one)
        precision_two_test = self.precision_per_class(self.clf_two)
        recall_two_test = self.recall_per_class(self.clf_two)
        f1_two_test = self.f1_per_class(self.clf_two)

        for klass in precision_one_test:
            precision_result =  precision_one_test[klass] < precision_two_test[klass]
            recall_result = recall_one_test[klass] < recall_two_test[klass]
            f1_result = f1_one_test[klass] < f1_two_test[klass]
            if precision_result or recall_result or f1_result:
                return False
        return True
        
    def cross_val_precision_per_class(self, clf, cv=3):
        precision_score = self.precision_score()
        precision_scorer = metrics.make_scorer(precision_score)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        precision = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            precision[klass] = precision_scorer(y_class, y_pred_class) 
        return precision

    def cross_val_recall_per_class(self, clf, cv=3):
        recall_score = self.recall_score()
        recall_scorer = metrics.make_scorer(recall_score)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        recall = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            recall[klass] = recall_scorer(y_class, y_pred_class)
        return recall

    def cross_val_f1_per_class(self, clf, cv=3):
        f1_score = self.f1_score()
        f1_scorer = metrics.make_scorer(f1_score)        
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        f1 = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            f1[klass] = f1_scorer(y_class, y_pred_class)
        return f1

    def cross_val_two_model_classifier_testing(self, cv=3):
        precision_one_test = self.cross_val_precision_per_class(self.clf_one, cv=cv)
        recall_one_test = self.cross_val_recall_per_class(self.clf_one, cv=cv)
        f1_one_test = self.cross_val_f1_per_class(self.clf_one, cv=cv)
        precision_two_test = self.cross_val_precision_per_class(self.clf_two, cv=cv)
        recall_two_test = self.cross_val_recall_per_class(self.clf_two, cv=cv)
        f1_two_test = self.cross_val_f1_per_class(self.clf_two, cv=cv)

        for klass in precision_one_test:
            precision_result =  precision_one_test[klass] < precision_two_test[klass]
            recall_result = recall_one_test[klass] < recall_two_test[klass]
            f1_result = f1_one_test[klass] < f1_two_test[klass]
            if precision_result or recall_result or f1_result:
                return False
        return True

    def cross_val_precision(self, clf, cv=3):
        precision_score = self.precision_score()
        precision_scorer = metrics.make_scorer(precision_score)        
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return precision_scorer(self.y, y_pred) 

    def cross_val_recall(self, clf, cv=3):
        recall_score = self.recall_score()
        recall_scorer = metrics.make_scorer(recall_score)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return recall_scorer(self.y, y_pred)

    def cross_val_f1(self, clf, cv=3):
        f1_score = self.f1_score()
        f1_scorer = metrics.make_scorer(f1_score)        
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return f1_scorer(self.y, y_pred)
        
    def cross_val_two_model_classifier_testing(self, cv=3):
        precision_one_test = self.cross_val_precision(self.clf_one, cv=cv)
        recall_one_test = self.cross_val_recall(self.clf_one, cv=cv)
        f1_one_test = self.cross_val_f1(self.clf_one, cv=cv)
        precision_two_test = self.cross_val_precision(self.clf_two, cv=cv)
        recall_two_test = self.cross_val_recall(self.clf_two, cv=cv)
        f1_two_test = self.cross_val_f1(self.clf_two, cv=cv)
        precision_result =  precision_one_test > precision_two_test
        recall_result = recall_one_test > recall_two_test
        f1_result = f1_one_test > f1_two_test
        if precision_result and recall_result and f1_result:
            return True
        else:
            return False
