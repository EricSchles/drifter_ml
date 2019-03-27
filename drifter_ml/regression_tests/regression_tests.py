from sklearn import metrics
import numpy as np
import time
from scipy import stats
from sklearn.model_selection import cross_validate, cross_val_predict

class RegressionTests():
    def __init__(self,
                 reg,
                 test_data,
                 target_name,
                 column_names):
        self.reg = reg
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]

    def get_test_score(self, cross_val_dict):
        return list(cross_val_dict["test_score"])

    def mse_cv(self, cv):
        mse = metrics.make_scorer(metrics.mean_squared_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(mse))
        return self.get_test_score(result)

    def _cross_val_anomaly_detection(scores, tolerance):
        avg = np.mean(scores)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviances_from_avg:
            if deviance > tolerance:
                return False
        return True

    def _cross_val_avg(self, scores, minimum_center_tolerance):
        avg = np.mean(scores)
        if avg < minimum_center_tolerance:
            return False
        return True

    def _cross_val_upper_boundary(self, scores, upper_boundary, cv=3):
        for score in scores:
            if score > upper_boundary:
                return False
        return True

    def cross_val_mse_anomaly_detection(self, tolerance, cv=3):
        scores = self.mse_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance)

    def cross_val_mse_avg(self, minimum_center_tolerance, cv=3):
        scores = self.mse_cv(cv)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_mse_upper_boundary(self, upper_boundary, cv=3):
        scores = self.mse_cv(cv)
        return self._cross_val_upper_boundary(scores, upper_boundary)
        
    def mse_upper_boundary(self, upper_boundary):
        y_pred = self.reg.predict(self.X)
        if metrics.mean_squared_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def mae_cv(self, cv):
        mse = metrics.make_scorer(metrics.median_absolute_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(mae))
        return self.get_test_score(result)
    
    def cross_val_mae_anomaly_detection(self, tolerance, cv=3):
        scores = self.mae_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance)

    def cross_val_mae_avg(self, minimum_center_tolerance, cv=3):
        scores = self.mae_cv(cv)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_mae_upper_boundary(self, upper_boundary, cv=3):
        scores = self.mae_cv(cv)
        return self._cross_val_upper_boundary(scores, upper_boundary)
    
    def mae_upper_boundary(self, upper_boundary):
        y_pred = self.reg.predict(self.X)
        if metrics.median_absolute_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def regression_testing(self, mse_upper_boundary, mae_upper_boundary):
        mse_test = self.mse_upper_boundary(mse_upper_boundary)
        mae_test = self.mae_upper_boundary(mae_upper_boundary)
        if mse_test and mae_test:
            return True
        else:
            return False

    def run_time_stress_test(self, performance_boundary):
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

class RegressionComparison():
    def __init__(self,
                 reg_one,
                 reg_two,
                 test_data,
                 target_name,
                 column_names):
        self.reg_one = reg_one
        self.reg_two = reg_two
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        
    def two_model_prediction_run_time_stress_test(self, performance_boundary):
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

    def cross_val_mse_result(self, reg, cv=3):
        y_pred = cross_val_predict(self.reg, self.X, self.y)
        return metrics.mean_squared_error(self.y, y_pred)
        
    def cross_val_mae_result(self, reg, cv=3):
        y_pred = cross_val_predict(self.reg, self.X, self.y)
        return metrics.median_absolute_error(self.y, y_pred)

    def mse_result(self, reg):
        y_pred = reg.predict(self.X)
        return metrics.mean_squared_error(self.y, y_pred)

    def mae_result(self, reg):
        y_pred = reg.predict(self.X)
        return metrics.median_absolute_error(self.y, y_pred)

    def cv_two_model_regression_testing(self):
        mse_one_test = self.cross_val_mse_result(self.reg_one)
        mae_one_test = self.cross_val_mae_result(self.reg_one)
        mse_two_test = self.cross_val_mse_result(self.reg_two)
        mae_two_test = self.cross_val_mae_result(self.reg_two)
        if mse_one_test < mse_two_test and mae_one_test < mae_two_test:
            return True
        else:
            return False

    def two_model_regression_testing(self):
        mse_one_test = self.mse_result(self.reg_one)
        mae_one_test = self.mae_result(self.reg_one)
        mse_two_test = self.mse_result(self.reg_two)
        mae_two_test = self.mae_result(self.reg_two)
        if mse_one_test < mse_two_test and mae_one_test < mae_two_test:
            return True
        else:
            return False
