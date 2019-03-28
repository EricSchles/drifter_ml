from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_validate, cross_val_predict
from functools import partial
from sklearn.model_selection import KFold
from sklearn.base import clone

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

    
# ToDo: reorganize this class into a bunch of smaller classes that inherit into a main class
class ClassificationTests(FixedClassificationMetrics):
    def __init__(self,
                 clf,
                 test_data,
                 target_name,
                 column_names):
        self.clf = clf
        self.test_data = test_data
        self.column_names = column_names
        self.target_name = target_name
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)

    def _get_per_class(self, y_true, y_pred, metric):
        class_measures = {klass: None for klass in self.classes}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, y_true[y_true == klass].index, axis=0)
            y_class = y_true[y_true == klass]
            class_measures[klass] = metric(y_class, y_pred_class)
        return class_measures

    def _per_class_cross_val(self, metric, cv, random_state=42):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        clf = clone(self.clf)
        scores = []
        for train, test in kfold.split(self.test_data):
            train_data = self.test_data.loc[train]
            test_data = self.test_data.loc[test]
            clf.fit(train_data[self.column_names], train_data[self.target_name])
            y_pred = clf.predict(test_data[self.column_names])
            y_true = test_data[self.target_name]
            y_true.index = list(range(len(y_true)))
            scores.append(self._get_per_class(y_true, y_pred, metric))
        return scores
            
    def get_test_score(self, cross_val_dict):
        return list(cross_val_dict["test_score"])
    
    # add cross validation per class tests
    def precision_cv(self, cv, average='binary'):
        precision_score = partial(self.precision_score, average=average)
        precision = metrics.make_scorer(precision_score)
        result =  cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(precision))
        return self.get_test_score(result)
    
    def recall_cv(self, cv, average='binary'):
        recall_score = partial(self.recall_score, average=average)
        recall = metrics.make_scorer(recall_score)
        result = cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(recall))
        return self.get_test_score(result)
    
    def f1_cv(self, cv, average='binary'):
        f1_score = partial(self.f1_score, average=average)
        f1 = metrics.make_scorer(f1_score)
        result = cross_validate(self.clf, self.X,
                              self.y, cv=cv,
                              scoring=(f1))
        return self.get_test_score(result)
        
    def _cross_val_anomaly_detection(self, scores, tolerance):
        avg = np.mean(scores)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviance_from_avg:
            if deviance > tolerance:
                return False
        return True

    def _cross_val_per_class_anomaly_detection(self, metric, tolerance, cv):
        scores_per_fold = self._per_class_cross_val(metric, cv)
        results = [] 
        for klass in self.classes:
            scores = [score[klass] for score in scores_per_fold]
            results.append(self._cross_val_anomaly_detection(scores, tolerance))
        return all(results)

    def cross_val_per_class_precision_anomaly_detection(self, tolerance,
                                                        cv=3, average='binary'):
        precision_score = partial(self.precision_score, average=average)
        return self._cross_val_per_class_anomaly_detection(precision_score,
                                                           tolerance, cv)

    def cross_val_per_class_recall_anomaly_detection(self, tolerance,
                                                     cv=3, average='binary'):
        recall_score = partial(self.recall_score, average=average)
        return self._cross_val_per_class_anomaly_detection(recall_score,
                                                           tolerance, cv)

    def cross_val_per_class_f1_anomaly_detection(self, tolerance,
                                                 cv=3, average='binary'):
        f1_score = partial(self.f1_score, average=average)
        return self._cross_val_per_class_anomaly_detection(f1_score,
                                                           tolerance, cv)

    def cross_val_precision_anomaly_detection(self, tolerance, cv=3, average='binary'):
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_anomaly_detection(scores, tolerance)
    
    def cross_val_recall_anomaly_detection(self, tolerance, cv=3, average='binary'):
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_anomaly_detection(scores, tolerance)
    
    def cross_val_f1_anomaly_detection(self, tolerance, cv=3, average='binary'):
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_anomaly_detection(scores, tolerance)

    def _cross_val_avg(self, scores, minimum_center_tolerance):
        avg = np.mean(scores)
        if avg < minimum_center_tolerance:
            return False
        return True

    def cross_val_precision_avg(self, minimum_center_tolerance, cv=3, average='binary'):
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_recall_avg(self, minimum_center_tolerance, cv=3, average='binary'):
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_f1_avg(self, minimum_center_tolerance, cv=3, average='binary'):
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def _cross_val_lower_boundary(self, scores, lower_boundary):
        for score in scores:
            if score < lower_boundary:
                return False
        return True
                      
    def cross_val_precision_lower_boundary(self, lower_boundary, cv=3, average='binary'):
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
        
    def cross_val_recall_lower_boundary(self, lower_boundary, cv=3, average='binary'):
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
        
    def cross_val_f1_lower_boundary(self, lower_boundary, cv=3, average='binary'):
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
    
    def cross_val_classifier_testing(self,
                                     precision_lower_boundary: float,
                                     recall_lower_boundary: float,
                                     f1_lower_boundary: float,
                                     cv=3, average='binary'):
        precision_test = self.cross_val_precision_lower_boundary(
            precision_lower_boundary, cv=cv, average=average)
        recall_test = self.cross_val_recall_lower_boundary(
            recall_lower_boundary, cv=cv, average=average)
        f1_test = self.cross_val_f1_lower_boundary(
            f1_lower_boundary, cv=cv, average=average)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def trimean(self, data):
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        median = np.median(data)
        return (q1 + 2*median + q3)/4

    def trimean_absolute_deviation(self, data):
        trimean = self.trimean(data)
        numerator = [abs(elem - trimean) for elem in data]
        return sum(numerator)/len(data)
        
    def describe_scores(self, scores, method):
        if method == "mean":
            return np.mean(scores), np.std(scores)
        elif method == "median":
            return np.median(scores), stats.iqr(scores)
        elif method == "trimean":
            return self.trimean(scores), self.trimean_absolute_deviation(scores)

    def _anomaly_detection(self, scores, tolerance, method):
        center, spread = self.describe_scores(scores, method)
        for score in scores:
            if score < center - (spread * tolerance):
                return False
        return True

    def spread_cross_val_precision_anomaly_detection(self, tolerance,
                                                     method="mean", cv=10, average='binary'):
        scores = self.precision_cv(cv, average=average)
        return self._anamoly_detection(scores, tolerance, method)
        
    def spread_cross_val_f1_anomaly_detection(self, tolerance,
                                              method="mean", cv=10, average='binary'):
        scores = self.f1_cv(cv, average=average)
        return self._anamoly_detection(scores, tolerance, method)
    
    def spread_cross_val_recall_anomaly_detection(self, tolerance,
                                                  method="mean", cv=3, average='binary'):
        scores = self.recall_cv(cv, average=average)
        return self._anamoly_detection(scores, tolerance, method)
        
    def spread_cross_val_classifier_testing(self,
                                            precision_lower_boundary: int,
                                            recall_lower_boundary: int,
                                            f1_lower_boundary: int,
                                            cv=10, average='binary'):
        precision_test = self.auto_cross_val_precision_lower_boundary(
            precision_lower_boundary, cv=cv, average=average)
        recall_test = self.auto_cross_val_recall_lower_boundary(
            recall_lower_boundary, cv=cv, average=average)
        f1_test = self.auto_cross_val_f1_lower_boundary(
            f1_lower_boundary, cv=cv, average=average)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def _per_class(self, y_pred, metric, lower_boundary):
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metric(y_class, y_pred_class) < lower_boundary[klass]:
                return False
        return True

    # potentially include hyper parameters from the model
    # algorithm could be stored in metadata
    # Todo: determine if still relevant ^
    def precision_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, self.precision_score, lower_boundary)

    def recall_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, self.recall_score, lower_boundary)
    
    def f1_lower_boundary_per_class(self, lower_boundary: dict):
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, self.f1_score, lower_boundary)
    
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
