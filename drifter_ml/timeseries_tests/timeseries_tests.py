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

class TimeSeriesTests:
    def __init__(self):
        pass

                 
