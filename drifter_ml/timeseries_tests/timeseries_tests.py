#from mlxtend.evaluate import permutation_test
from statsmodels.tsa import stattools
from statsmodels.stats import diagnstic
import warnings
warnings.filterwarnings("ignore")

class TimeSeriesTests:
    def __init__(self, data):
        self.data = data

    def unit_root(self, column, pvalue_tolerance=0.05):
        """
        Unit root test, tests whether or not your time series is
        stationary.  The null hypothesis is that it is not.
        The alternative hypothesis is that it is.  So if the pvalue is
        less than the threshold, the series is stationary
        
        if True then stationary
        if False than not stationary
        """
        result = stattools.adfuller(self.data[column])
        if result[1] < pvalue_tolerance:
            return True
        return False

    def kpss(self, column, pvalue_tolerance=0.05):
        """
        KPSS also tests for stationarity, here the null hypothesis
        is that the series is stationary, so a pvalue above the tolerance level
        means it's stationary.  A pvalue below means the alternative hypothesis
        is true and it is not.

        if True then stationary
        if False than not stationary
        """
        result = stattools.kpss(self.data[column])
        if result[1] > pvalue_tolerance:
            return True
        return False

    def coint(self, column_one, column_two, pvalue_tolerance=0.05):
        """
        tests whether or not two series are cointegrated.
        Cointegration is similar to correlation, because there
        is a dependence relationship in two random series
        
        if True than column_one not cointegrated with column_two
        if False than column_one cointegrated with column_two
        """
        result = stattools.coint(self.data[column_one], self.data[column_two])
        if result[1] > pvalue_tolerance:
            return True
        return False
        
    def bds(self, column_one, pvalue_tolerance=0.05):
        """
        tests if the column is independent and identically distributed
        
        if True variable is iid
        if False variable is not
        """
        result = stattools.bds(self.data[column_one])
        if result[1] > pvalue_tolerance:
            return True
        return False

    def q_stat(self, column_one, pvalue_tolerance=0.05):
        """
        tests for auto correlation in the series, this is similiar to
        serial correlation, as outlied here: 
        http://gauss.stat.su.se/gu/e/slides/Lectures%208-13/Autocorrelation.pdf
        
        the null hypothesis is no autocorrelation
        the alternative hypothesis is autocorrelation exists.
        
        if True variable is not autocorrelated
        if False variable is autocorrelated
        """
        autocorrelation_coefs = stattools.acf(self.data[column_one])
        result = stattools.q_stat(autocorrelation_coefs)
        if result[1] > pvalue_tolerance:
            return True
        return False

    
# ToDo:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.unitroot_adf.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html

