'''
The goal of this model is to test for model drift:  Does the model behave the same
way over time?

Is the model and data consistent over time?

We can think of this through the following questions, 

* do the same inputs produce the same outputs, over time?
* how sensitive is the model to input data?
* what is the distribution of predictions over time?
* what are the marginal distributions of the data over time?
* As the marginal distributions change, how much does the distribution of predictions change, over time?
* how stable is the distribution of predictions over time? (for regression)
* how stable are the percentages per class of the predictions over time? (for classification)
* how likely are certain predictions over time? (for classification)
* how likely are certain ranges of predictions over time? (for regression)

* how much data do we expect to be misclassified over time? (for classification)
  * precision
  * recall
  * f1 score
* how much error do we expect over time? (for regression)
  * mean squared error
  * median absolute error
  * trimean absolute error
* how many outliers do we expect in the data over time? (using various techniques)
* how likely is it the data is drawn from the same distribution over a given time frame? (using distribution similarity tests)
* how sensitive is the model to changes in each marginal variable over time? (regression and classification) IE, if we change each variable while holding all others constant, how many values do we need to change to produce a significant change in the prediction (significant increase in the output for regression) or change of class for classification?  
* how sensitive is the model to the marginal effects of n variables? (with the above set up) where n varies from 1 to the total number of variables in the data
* how do various feature selection algorithms change on the data over time? aka which features are statistically significant over time?
* how much of the data is missing over time?
'''
from backtester import metrics as bt_metrics
import pandas as pd
import datetime

class TimeSeriesClassificationTests:
    """
    The general goal of this class is to test 
    classification algorithms over time.
    The class expects the following parameters:
    
    * descriptors : arraylike
      A set of descriptions of a model.  This ought to
      be a classification metric like precision, recall, or
      f1-score or a loss like log loss.

    * timestamps : arraylike
      A set of timestamps associated with the descriptors.
      this will be important for some of the metrics used.
      Each element should be of time datetime.datetime.
    
    The way in which classification algorithms is assessed via
    hypothesis tests and time series metrics. The time series
    metrics come to us from backtester, another framework I developed.
    Each timeseries metric is standard where the expectation is
    that data is compared against a forecast.  
    A simple moving average is used for the forecast model to make
    sure the only thing we are trying to capture is how much the model
    has changed recently.
    
    For this reason, the number of lag periods is very important. If you
    set this number too low, you may think everything is fine, when in fact
    things are actually changing quiet rapidly.  If you set the number of lags
    too long, then you may capture bugs from the last anomaly, and thus won't
    capture the next.
    
    A good rule of thumb is to set the number of lags for a week, assuming everything
    has been fine.  And set it for 5 periods after the last bug, to assess normality.
    
    It may make sense to initialize multiple instances of the class, if
    you want to capture things at different levels of granularity.
    """
    def __init__(self, descriptors, timestamps, lags=10):
        """
        Initialize the series.

        Args:
            self: (todo): write your description
            descriptors: (todo): write your description
            timestamps: (int): write your description
            lags: (int): write your description
        """
        self.descriptors = discriptors
        self.timestamps = timestamps
        self.lags = lags
        self.series = self._generate_series()
        
    def _generate_series(self):
        """
        Generate a pandas. series. series.

        Args:
            self: (todo): write your description
        """
        return pd.Series(
            data = self.descriptors,
            index = self.timestamps
        )

    def _apply_metric(self, metric, forecast_start, max_error):
        """
        Perform the prediction.

        Args:
            self: (todo): write your description
            metric: (str): write your description
            forecast_start: (todo): write your description
            max_error: (int): write your description
        """
        y_true = series[forecast_start:]
        y_pred = self.series.rolling(window=self.lags).mean()
        y_pred = y_pred[forecast_start:]
        error = metric(
            y_true, y_pred
        )
        return error < max_error

    def root_mean_squared_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The root mean squared error is a standard metric for 
        assessing error in a regression problem.  It lends itself
        naturally to the forecast context because of its application
        of a euclidean metric as well as taking of the average.
        
        An average is especially advantegous due to its sensitivity
        to outliers.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the root mean squared error of 
        the forecast and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.root_mean_squared_error,
            forecast_start, max_error
        )
        
    def normalized_root_mean_squared_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The normalized root mean squared error takes into account scale.
        It is not recommended that the normalized root mean squared error
        be used if your descriptor is a score, since those are already bounded
        between (0.0, 1.0).  If you are dealing with a loss function, then
        the normalized root mean squared error may be advantegous as sense of
        scale is removed.
        
        Since there is no standard convention for how to normalize the choice
        of max - min of the observations is used as a choice for normalization.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the normalized root mean squared error of 
        the forecast and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.normalized_root_mean_squared_error,
            forecast_start, max_error
        )

    def mean_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        Perhaps the most naive metric I could think of, mean error
        is simply the average error of the forecast against the
        observations.
        
        As a result, this measure will be sensitive to outliers, which may
        be advantegous for assessing deviance quickly and obviously.

        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the mean error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.mean_error,
            forecast_start, max_error
        )

    def mean_absolute_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        Perhaps one of the most naive metrics out there, mean absolute error
        is simply the average of the absolute value of the error of the forecast against the
        observations.
        
        It ought to be the same as mean error, because score functions are bounded to the
        range (0.0, 1.0) and loss functions should never be negative.  That said
        it is always possible something went wrong.  It therefore might be useful
        to run mean absolute error and mean error with the same parameters.  If
        one passes and the other fails, this will be a good signal that something is
        wrong with your set up.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the mean absolute error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.mean_absolute_error,
            forecast_start, max_error
        )

    def median_absolute_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The median absolute error is an interesting metric to look at.  It ignores outliers,
        so it may be used as an expectation of normalcy without the outliers.  Comparing
        median absolute error and mean absolute error might give a sense of how much outliers
        are effecting centrality.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the median absolute error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.median_absolute_error,
            forecast_start, max_error
        )

    def variance_absolute_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The variance absolute error gives us a sense of the variance in our error.  This way
        we can directly interrogate variability in our absolute error.  And we can set boundaries
        for the maximum boundary on deviances from our forecast.

        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the variance absolute error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.median_absolute_error,
            forecast_start, max_error
        )
    
    def mean_squared_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The mean squared error is a canonical measure of error.  It overstates large deviations
        of individual examples while marginalizing the effect size of any deviances of deviations
        smaller than one.  Because the mean is used, large values are overstated, thus individual
        large deviations will tend to become apparent.  For the mean squared error to be small,
        therefore no extreme deviances must exist.  However relatively small deviances across
        many or even all samples will be understated.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the mean squared error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.mean_squared_error,
            forecast_start, max_error
        )

    def mean_squared_log_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The mean squared log error is a variant on mean squared error.  Mean squared log error
        measures the relative difference between the true and predicted values.
        It over penalizes underestimates, cases where the predicted value is less than
        the true value, more than it penalizes overestimates, cases where the predicted
        value is more than the true value.  This is because it's a MSLE is a ratio of the two.
        
        This measure is especially useful if you want to check if your prediction is smaller
        than your actual timeseries.  Therefore it is very useful for accuracy and less
        useful for error metrics.
        
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the mean squared error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.mean_squared_log_error,
            forecast_start, max_error
        )

    def root_mean_squared_log_error(self, forecast_start: datetime.datetime, max_error: float) -> bool:
        """
        The root mean squared log error is a variant on mean squared error.  
        Root mean squared log error measures the relative difference between 
        the true and predicted values. It over penalizes underestimates, cases 
        where the predicted value is less than the true value, more than it 
        penalizes overestimates, cases where the predicted value is more than the true value.  
        This is because it's a RMSLE is a ratio of the two.  However unlike the MSLE
        by taking the root the penalization is diminished making this closer in measure
        to something like the mean squared error in terms of direction.
                
        Parameters
        ----------
        * forecast_start : datetime.datetime
          The starting timestamp to begin the forecast.
          Observations of the descriptor after the start time will be checked.
          Special care should be given when choosing the start forecast.
        
        * max_error: float
          The maximum allowed error or tolerance of the forecast.
          If we are dealing with a score function like f1-score
          it is imperative that we set max_error below 1.0.
        
        Return
        ------
        True if the mean squared error of the forecast 
        and actual error is below the max_error.
        False otherwise
        """
        return self._apply_metric(
            bt_metric.root_mean_squared_log_error,
            forecast_start, max_error
        )


# iqr_absolute_error
# geometric_mean_absolute_error
# mean_percentage_error
# mean_absolute_percentage_error
# median_absolute_percentage_error
# symmetric_mean_absolute_percentage_error
# symmetric_median_absolute_percentage_error
# mean_arctangent_absolute_percentage_error
# mean_absolute_scaled_error
# normalized_absolute_error
# normalized_absolute_percentage_error
# root_mean_squared_percentage_error
# root_median_squared_percentage_error
# root_mean_squared_scaled_error
# integral_normalized_root_squared_error
# root_relative_squared_error
# mean_relative_error
# relative_absolute_error
# mean_relative_absolute_error
# median_relative_absolute_error
# geometric_mean_relative_absolute_error
# mean_bounded_relative_absolute_error
# unscaled_mean_bounded_relative_absolute_error
# mean_directional_accuracy
 
