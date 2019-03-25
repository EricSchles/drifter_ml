#from mlxtend.evaluate import permutation_test
from statsmodels.tsa import stattools
from statsmodels.stats import diagnstic
import warnings
from collections import namedtuple
warnings.filterwarnings("ignore")

class TimeSeriesTests:
    def __init__(self):
        pass

    def ad_fuller_test(self, timeseries):
        result = stattools.adfuller(timeseries)
        AdFullerResult = namedtuple('AdFullerResult', 'statistic pvalue')
        return AdFullerResult(result[0], result[1])
    
    def kpss(self, timeseries):
        result = stattools.kpss(timeseries)
        KPSSResult = namedtuple('KPSSResult', 'statistic pvalue')
        return KPSSResult(result[0], result[1])

    def cointegration(self, timeseries_one, timeseries_two):
        result = stattools.coint(timeseries_one, timeseries_two)
        CointegrationResult = namedtuple('CointegrationResult', 'statistic pvalue')
        return CointegrationResult(result[0], result[1])

    def bds(self, timeseries):
        result = stattools.bds(timeseries)
        BdsResult = namedtuple('BdsResult', 'statistic pvalue')
        return BdsResult(result[0], result[1])
        
    def q_stat(self, timeseries):
        autocorrelation_coefs = stattools.acf(timeseries)
        result = stattools.q_stat(autocorrelation_coefs)
        QstatResult = namedtuple('QstatResult', 'statistic pvalue')
        return QstatResult(result[0], result[1])

    
# ToDo:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.unitroot_adf.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html

