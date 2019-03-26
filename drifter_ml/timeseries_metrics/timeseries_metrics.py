import numpy as np
from sklearn import metrics
from scipy.stats import mstats

class ForecastMetrics:
    def __init__(self):
        pass
    
    @staticmethod
    def unscaled_mean_bounded_relative_absolute_error(y_true, y_pred):
        """
        Unscaled Mean Bounded Relative Absolute Error
        Formula taken from:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
        @y_true - Y[i]
        @y_pred - F[i]
        """
        numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx]) for idx, elem in enumerate(series_one)]
        final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                        for idx in range(len(denominator))]
        mbrae = np.mean(final_series)
        return mbrae/(1-mbrae)

    @staticmethod
    def mean_bounded_relative_absolute_error(y_true, y_pred):
        """
        Mean Bounded Relative Absolute Error
        Formula taken from:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
        @y_true - Y[i]
        @y_pred - F[i]
        """
        numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx]) for idx, elem in enumerate(series_one)]
        final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                        for idx in range(len(denominator))]
        return np.mean(final_series)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def mean_relative_absolute_error(y_true, y_pred):
        """
        formula comes from: 
        http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
        """
        numerator = [abs(elem - y_pred[idx])
                     for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]    
        return np.mean([
            numerator[i]/denominator[i] for i in range(len(numerator))])

    @staticmethod
    def median_relative_absolute_error(y_true, y_pred):
        """
        formula comes from: 
        http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
        """
        numerator = [abs(elem - y_pred[idx])
                     for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]    
        return np.median([
            numerator[i]/denominator[i] for i in range(len(numerator))])

    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        numerator = [abs(y_pred[idx] - elem) for idx, elem in enumerate(y_true)]
        denominator = [abs(elem) + abs(y_pred[idx]) for idx, elem in enumerate(y_true)]
        denominator = [elem/2 for elem in denominator]
        result = np.mean([numerator[i]/denominator[i] for i in range(len(numerator))])
        return result * 100

    @staticmethod
    def symmetric_median_absolute_percentage_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        numerator = [abs(y_pred[idx] - elem) for idx, elem in enumerate(y_true)]
        denominator = [abs(elem) + abs(y_pred[idx]) for idx, elem in enumerate(y_true)]
        denominator = [elem/2 for elem in denominator]
        result = np.median([numerator[i]/denominator[i] for i in range(len(numerator))])
        return result * 100

    @staticmethod
    def mean_absolute_scaled_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        """
        numerator = sum([abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)])
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = sum([abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)])
        coeficient = len(y_true)/(len(y_true)-1)
        return numerator/(coeficient * denominator)

    @staticmethod
    def geometric_mean_relative_absolute_error(y_true, y_pred):
        numerator = [abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]
        return mstats.gmean([numerator[i]/denominator[i] for i in range(len(numerator))])

