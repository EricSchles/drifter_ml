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

    def trimean(self, data):
        """
        I'm exposing this as a public method because
        the trimean is not implemented in enough packages.
        
        Formula:
        (25th percentile + 2*50th percentile + 75th percentile)/4
        
        Parameters
        ----------
        data : array-like
          an iterable, either a list or a numpy array

        Returns
        -------
        the trimean: float
        """
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        median = np.median(data)
        return (q1 + 2*median + q3)/4

    def trimean_absolute_deviation(self, data):
        """
        The trimean absolute deviation is the
        the average distance from the trimean.
        
        Parameters
        ----------
        data : array-like
          an iterable, either a list or a numpy array

        Returns
        -------
        the average distance to the trimean: float
        """
        trimean = self.trimean(data)
        numerator = [abs(elem - trimean) for elem in data]
        return sum(numerator)/len(data)

    def describe_scores(self, scores, method):
        """
        Describes scores.
        
        Parameters
        ----------
        scores : array-like
          the scores from the model, as a list or numpy array
        method : string
          the method to use to calculate central tendency and spread
        
        Returns
        -------
        Returns the central tendency, and spread
        by method.
        
        Methods:
        mean:
        * central tendency: mean
        * spread: standard deviation
        
        median:
        * central tendency: median
        * spread: interquartile range
        
        trimean:
        * central tendency: trimean
        * spread: trimean absolute deviation
        """
        if method == "mean":
            return np.mean(scores), np.std(scores)
        elif method == "median":
            return np.median(scores), stats.iqr(scores)
        elif method == "trimean":
            return self.trimean(scores), self.trimean_absolute_deviation(scores)

    def mae_cv(self, cv):
        """
        This method performs cross-validation over median absolute error.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform

        Returns
        -------
        Returns a scores of the k-fold median absolute error.
        """

        mae = metrics.make_scorer(metrics.median_absolute_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(mae))
        return self.get_test_score(result)
    
    def mse_cv(self, cv):
        """
        This method performs cross-validation over mean squared error.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform

        Returns
        -------
        Returns a scores of the k-fold mean squared error.
        """
        mse = metrics.make_scorer(metrics.mean_squared_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(mse))
        return self.get_test_score(result)

    def trimean_squared_error(self, y_true, y_pred,
                              sample_weight=None,
                              multioutput='uniform_average'):
        output_errors = self.trimean((y_true - y_pred) ** 2)
        return self.trimean(output_errors)

    def trimean_absolute_error(self, y_true, y_pred,
                               sample_weight=None,
                               multioutput='uniform_average'):
        output_errors = self.trimean(abs(y_true - y_pred))
        return self.trimean(output_errors)

    def tse_cv(self, cv):
        """
        This method performs cross-validation over trimean squared error.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform

        Returns
        -------
        Returns a scores of the k-fold trimean squared error.
        """
        tse = metrics.make_scorer(self.trimean_squared_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(tse))
        return self.get_test_score(result)

    def tae_cv(self, cv):
        """
        This method performs cross-validation over trimean absolute error.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform

        Returns
        -------
        Returns a scores of the k-fold trimean absolute error.
        """
        tse = metrics.make_scorer(self.trimean_absolute_error)
        result = cross_validate(self.reg, self.X,
                                self.y, cv=cv,
                                scoring=(tse))
        return self.get_test_score(result)

    def _cross_val_anomaly_detection(self, scores, tolerance, method='mean'):
        avg, _ = self.describe_scores(scores, method)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviance_from_avg:
            if deviance > tolerance:
                return False
        return True

    def _cross_val_avg(self, scores, maximum_center_tolerance, method='mean'):
        avg, _ = self.describe_scores(scores, method)
        if avg > maximum_center_tolerance:
            return False
        return True

    def _cross_val_upper_boundary(self, scores, upper_boundary):
        for score in scores:
            if score > upper_boundary:
                return False
        return True

    def cross_val_tse_anomaly_detection(self, tolerance, cv=3, method='mean'):
        scores = self.tse_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance, method=method)

    def cross_val_tse_avg(self, minimum_center_tolerance, cv=3, method='mean'):
        scores = self.tse_cv(cv)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_tse_upper_boundary(self, upper_boundary, cv=3):
        scores = self.tse_cv(cv)
        return self._cross_val_upper_boundary(scores, upper_boundary)
        
    def tse_upper_boundary(self, upper_boundary):
        y_pred = self.reg.predict(self.X)
        if self.trimean_squared_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def cross_val_tae_anomaly_detection(self, tolerance, cv=3, method='mean'):
        scores = self.tae_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance, method=method)

    def cross_val_tae_avg(self, minimum_center_tolerance, cv=3, method='mean'):
        scores = self.tae_cv(cv)
        return self._cross_val_avg(scores, minimum_center_tolerance)

    def cross_val_tae_upper_boundary(self, upper_boundary, cv=3):
        scores = self.tae_cv(cv)
        return self._cross_val_upper_boundary(scores, upper_boundary)
        
    def tae_upper_boundary(self, upper_boundary):
        y_pred = self.reg.predict(self.X)
        if self.trimean_absolute_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def cross_val_mse_anomaly_detection(self, tolerance, cv=3, method='mean'):
        scores = self.mse_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance, method=method)

    def cross_val_mse_avg(self, minimum_center_tolerance, cv=3, method='mean'):
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
    
    def cross_val_mae_anomaly_detection(self, tolerance, cv=3, method='mean'):
        scores = self.mae_cv(cv)
        return self._cross_val_anomaly_detection(scores, tolerance, method=method)

    def cross_val_mae_avg(self, minimum_center_tolerance, cv=3, method='mean'):
        scores = self.mae_cv(cv)
        return self._cross_val_avg(scores, minimum_center_tolerance, method=method)

    def cross_val_mae_upper_boundary(self, upper_boundary, cv=3):
        scores = self.mae_cv(cv)
        return self._cross_val_upper_boundary(scores, upper_boundary)
    
    def mae_upper_boundary(self, upper_boundary):
        y_pred = self.reg.predict(self.X)
        if metrics.median_absolute_error(self.y, y_pred) > upper_boundary:
            return False
        return True

    def upper_bound_regression_testing(self,
                                       mse_upper_boundary,
                                       mae_upper_boundary,
                                       tse_upper_boundary,
                                       tae_upper_boundary):
        mse_test = self.mse_upper_boundary(mse_upper_boundary)
        mae_test = self.mae_upper_boundary(mae_upper_boundary)
        tse_test = self.tse_upper_boundary(tse_upper_boundary)
        tae_test = self.tae_upper_boundary(tae_upper_boundary)
        if mse_test and mae_test and tse_test and tae_test:
            return True
        else:
            return False

    def run_time_stress_test(self, sample_sizes, max_run_times):
        for index, sample_size in enumerate(sample_sizes):
            max_run_time = max_run_times[index]
            data = self.X.sample(sample_size, replace=True)
            start_time = time.time()
            self.reg.predict(data)
            model_run_time = time.time() - start_time
            if model_run_time > max_run_time:
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
        
    def two_model_prediction_run_time_stress_test(self, sample_sizes):
        for sample_size in sample_sizes:
            data = self.X.sample(sample_size, replace=True)
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
        y_pred = cross_val_predict(reg, self.X, self.y)
        return metrics.mean_squared_error(self.y, y_pred)
        
    def cross_val_mae_result(self, reg, cv=3):
        y_pred = cross_val_predict(reg, self.X, self.y)
        return metrics.median_absolute_error(self.y, y_pred)

    def mse_result(self, reg):
        y_pred = reg.predict(self.X)
        return metrics.mean_squared_error(self.y, y_pred)

    def mae_result(self, reg):
        y_pred = reg.predict(self.X)
        return metrics.median_absolute_error(self.y, y_pred)

    def cv_two_model_regression_testing(self, cv=3):
        mse_one_test = self.cross_val_mse_result(self.reg_one, cv=cv)
        mae_one_test = self.cross_val_mae_result(self.reg_one, cv=cv)
        mse_two_test = self.cross_val_mse_result(self.reg_two, cv=cv)
        mae_two_test = self.cross_val_mae_result(self.reg_two, cv=cv)
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
