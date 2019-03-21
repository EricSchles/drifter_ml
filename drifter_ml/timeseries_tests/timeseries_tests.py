#from mlxtend.evaluate import permutation_test
from statsmodels.stats import diagnostic
import warnings
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

class TimeSeriesTests:
    def __init__(self, timeseries, ar_values, d_values, ma_values, is_date=False, is_datetime=False, is_time=False):
        self.timeseries = timeseries
        if is_date:
            self.timeseries.index = pd.to_datetime(timeseries["date"])
            self.timeseries.drop("date", inplace=True)
        if is_datetime:
            self.timeseries.index = pd.to_datetime(timeseries["datetime"])
            self.timeseries.drop("datetime", inplace=True)
        if is_time:
            self.timeseries.index = timeseries["time"]
            self.timeseries.drop("time", inplace=True)
        self.ar_values = ar_values
        self.d_values = d_values
        self.ma_values = ma_values
        self.model = self._evaluate_models()
        
    # evaluate an ARIMA model for a given order (p,d,q)
    def _evaluate_arima_model(self, X, arima_order):
        # prepare training dataset
        train, test, _, _ = train_test_split(X, np.zeros(X.shape[0]))
        history = list(train)
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        error = mean_squared_error(test, predictions)
        return error

    # evaluate combinations of p, d and q values for an ARIMA model
    def _evaluate_models(self):
        dataset = self.timeseries.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score = mse
                            best_order = order
                    except:
                        continue
        model = ARIMA(dateset, order=best_order)
        model_fit = model.fit(disp=0)
        return model_fit

    def acorr_breusch_godfrey(self, nlags=None):
        return diagnostic.acorr_breusch_godfrey(self.model, nlags=nlags)

    def unitroot_adf(self, maxlag=None, trendorder=0, autolag='AIC', store=False):
        return diagnostic.unitroot_adf(self.timeseries,
                                       maxlag=maxlag,
                                       trendorder=trendorder,
                                       autolag=autolag,
                                       store=store)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)


https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.unitroot_adf.html
https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html
https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html

# x = np.array([1, 2, 3, 4, 5, 6])
# y = np.array([2, 4, 1, 5, 6, 7])

# print('Observed pearson R: %.2f' % np.corrcoef(x, y)[1][0])


# p_value = permutation_test(x, y,
#                            method='exact',
#                            func=lambda x, y: np.corrcoef(x, y)[1][0],
#                            seed=0)
# print('P value: %.2f' % p_value)
