########################
Designing your own tests
########################

Before we jump into the API and all the premade tests that have been written to make your life easier, let's talk about a process for designing your own machine learning tests.  The reason for doing this is important, machine learning testing is not like other software engineering tests.  That's because software engineering tests are deterministic, like software engineering code ought to be.  However, when you write tests for your data or your machine learning model, you need to account for the probabilistic nature of the code you are writing.  The goal, therefore is much more fuzzy.  But the process defined below should help you out.

############################################
It's About Proving Or Disproving Assumptions
############################################

There are a standard set of steps to any machine learning project:

1. Exploratory Analysis
2. Data Cleaning
3. Model Evaluation
4. Productionalizing The Model
5. Monitoring The Model

Machine learning tests are really about monitoring, but the big open question is, what do you monitor?  

Monitoring the steps you took in 1-3 above, gives at least a base line.  There will likely be other things to account for and monitor once you go into production, but what you've found in evaluation will likely be helpful later.  So that should inform your first set of tests.

######################
Data Monitoring Tests
######################

Specifically, we can monitor the data by:

* checking to see if any descriptive statistics you found have changed substantially
* checking to see if current data is correlated with previous data per column
* checking to see if columns that were correlated or uncorrelated in past data remain that way
* checking to see if the number of clusters in the data has changed in a meaningful way
* checking to see whether the number of missing values stays consistent between new and old data, 
* checking to see certain monotonicity requirements between columns remain consistent

It is an imperative to model the data because your model is merely a function of your data.  If your data is bad or changes in some important way, your model will be useless.  Also, there may be more measures you used to evaluate the data and those may become important features of whatever model you build later on.  Therefore, making sure your data continues to follow the trends found previously may be of great import.  Otherwise, your model might be wrong and you'd never know it.  

#######################
Model Monitoring Tests
#######################

Additionally, we can monitor the model itself:

* checking to see if the model meets all metric requirements as specified by the business use-case
* checking to see if the model does better than some other test model on all measures of interest

########################
System Monitoring Tests
########################

Finally, there are also traditional tests one should run:

* making sure the serialized model exists where expected
* making sure the data exists where expected
* making sure data can flow into the system, to the model and through it
* making sure the new data matches the types you expect
* making sure the model produces the types you expect
* making sure new models can be deployed to the model pipeline
* making sure the model can perform well under load
* making sure the data can flow through fast enough to reach the model at ingress and egress

These three classes of machine learning system evaluation form a minimal reference set for monitoring such a system.  There are likely more tests you'll need to write, but again just to outline the process in clear terms:

1. Look at what you wrote when you did exploratory analysis and data cleaning, turn those into tests to make sure your data stays that way, as long as it's supposed to

2. Look at how your model performed on test and training data, turn those evaluation measures into tests to make sure your model performs as well in production

3. Make sure everything actually goes from point A (the start of your system) to point B (the end of your system).

#########################
Fairness Monitoring Tests
#########################

There is a fourth class of tests that are unclear regarding the ethical nature of the algorithm you are building.  These tests are unfortunately poorly defined at the present moment and very context specific, so all that can be offered is an example of what one might do:

Suppose you worked for a bank and were writing a piece of software that determined who gets a loan.  Assuming a fair system folks from all races, genders, ages would get loans at a similar rate or would perhaps not be rejected due to race, gender, age or other factors.

If when accounting for some protected variable like race, gender, or age your algorithm does something odd compared to when not accounting for race, gender, or age then your algorithm may be biased.  

However, this field of research is far from complete.  There are some notions of testing for this, at the present moment they appear to be in need of further research and analysis.  However, if possible, one should account for such a set of tests if possible, to ensure your algorithm is fair, unbiased and treats all individuals equally and fairly.