####################
Classification Tests
####################

The goal of the following set of tests is to accomplish some monitoring goals:

1. Establish baselines for model performance in production per class

2. Establish maximum processing time for various volumes of data, through the statistical model

3. Ensure that the current model in production is the best available model according to a set of predefined measures

Let's look at each of these classes of tests now.


Lower Bound Classification Measures
===================================

Each of the following examples ensures that your classifier meets a minimum criteria, which should be decided based on the need of your use-case.  One simple way to do this is to define failure by how many dollars it will cost you.  

Precision, Recall and F1 score are great tools for ensuring your classifier optimizes for minimal misclassification, however you define it.  

That is why they are basis of the set of tests found below.


Classifier Test Example - Model Metrics
=======================================

Suppose you had the following model::

	from sklearn import tree
	import pandas as pd
	import numpy as np
	import joblib

	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    if a + b + c > 11:
	        target = 1
	    else:
	        target = 0
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	clf = tree.DecisionTreeClassifier()
	X = df[["A", "B", "C"]]
	clf.fit(X, df["target"])
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")

We could write the following set of tests to ensure this model does well::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_precision():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    classes = list(df.target.unique())
	    assert test_suite.precision_lower_boundary_per_class(
	        {klass: 0.9 for klass in classes}
	    )

	def test_recall():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    classes = list(df.target.unique())
	    assert test_suite.recall_lower_boundary_per_class(
	        {klass: 0.9 for klass in classes}
	    )

	def test_f1():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    classes = list(df.target.unique())
	    assert test_suite.f1_lower_boundary_per_class(
	        {klass: 0.9 for klass in classes}
	    )


Or you could simply write one test for all three::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_precision_recall_f1():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    classes = list(df.target.unique())
	    assert test_suite.classifier_testing_per_class(
	        {klass: 0.9 for klass in classes},
	        {klass: 0.9 for klass in classes},
	        {klass: 0.9 for klass in classes}
	    )

Regardless of which test you choose, you get complete flexibility to ensure your model always meets the minimum criteria so that your costs are minimized, given constraints.


Classifier Test Example - Model Speed
=====================================

Additionally, you can test to ensure your classifier performs, even under load.  Assume we have the same model as before::

	from sklearn import tree
	import pandas as pd
	import numpy as np
	import joblib

	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    if a + b + c > 11:
	        target = 1
	    else:
	        target = 0
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	clf = tree.DecisionTreeClassifier()
	X = df[["A", "B", "C"]]
	clf.fit(X, df["target"])
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")

Now we test to ensure the model predicts new labels within our constraints::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_precision_recall_f1_speed():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
    	sample_sizes = [i for i in range(100, 1000, 100)]
    	max_run_times = [100 for _ in range(len(sample_sizes))]
	    assert test_suite.run_time_stress_test(
	        sample_sizes, max_run_times
	    )

This test ensures that from 1 to 100000 elements, the model never takes longer than 10 seconds.  

Cross Validation Based Testing
==============================

In the last section we asked questions of our model with respect to a lower boundary, both of various model measures as well as speed measurement in seconds.  Now armed with cross validation we can ask questions about sections of our dataset, to ensure that the measures we found were an accurate representation across the dataset, rather than one global metric across the entire dataset.  Just to make sure we are all on the same page, cross validation breaks the dataset into unique samples and then each sample is used as the test sample, all other samples are used as training, the score for each validation sample is recorded and then the model is discarded.  For more information and a detailed introduction see https://machinelearningmastery.com/k-fold-cross-validation/.  

The advantage of checking our model in this way is now it is less likely that the model is just memorizing the training data and will actually scale to other examples.  This happens because the model scores are tested on a more limited dataset and also because "k" samples, the tuning parameter in cross validation, are tested to ensure the model performance is consistent.  

This also yields some advantages for testing, because now we can verify that our lower boundary precision, recall or f1 score is true across many folds, rather than some global lower bound which may not be true on some subset of the data.  This gives us more confidence in our models overall efficacy, but also requires that we have enough data to ensure our model can learn something.  

Sadly I could find no good rules of thumb but I'd say less than you need at least something like 1000 data points per fold at least, and it's probably best to never go above 20 folds unless your dataset is truly massive, like in the gigabytes.


Classifier Test Example - Cross Validation Lower Bound Precision
================================================================

This example won't be that different from what you've seen before, except now we can tune on the number of folds to include.  Let's spice things up by using a keras classifier instead of a scikit learn one::

	from keras.models import Sequential
	from keras.layers import Dense
	from keras.wrappers.scikit_learn import KerasClassifier
	import pandas as pd
	import numpy as np
	import joblib

	# Function to create model, required for KerasClassifier
	def create_model():
	    # create model
	    model = Sequential()
	    model.add(Dense(12, input_dim=3, activation='relu'))
	    model.add(Dense(8, activation='relu'))
	    model.add(Dense(1, activation='sigmoid'))
	    # Compile model
	    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	    return model

	# fix random seed for reproducibility
	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    if a + b + c > 11:
	        target = 1
	    else:
	        target = 0
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	# split into input (X) and output (Y) variables
	# create model
	clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
	X = df[["A", "B", "C"]]
	clf.fit(X, df["target"])
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")

Now that we have the model and data saved, let's write the test::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_cv_precision_lower_boundary():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    lower_boundary = 0.9
	    test_suite.cross_val_precision_lower_boundary(
	    	lower_boundary
	    )

There are a few things to notice here:

1. The set up didn't change - we train the model the same way, we store the model the same way, we pass the model in the same way.

2. We aren't specifying percision per class - we will see examples of tests like that below, but because of the added stringency of limiting our training set, as well as training it across several samples of the dataset, sometimes called folds, we now don't need to specify as much granularity.  What we are really testing here is somewhat different - we want to make sure no samples of the dataset form significantly worse than the average.  What we are really looking for is anomalous samples of the data, that the model does much worse on.  Because any training set is just a sample, if a given subsample does much worse than others, then we need to ask the question - is this given subsample representative of a pattern we may see in the future?  Is it truly an anamoly?  If it's not, that's usually a strong indicator that our model needs some work.

Classifier Test Example - Cross Validation Average
===================================================

In the above example we test to ensure that none of the folds fall below a precision of 0.9 per fold.  But what if we only care if one of the folds does significantly worse than the others?  But don't actually care if all the folds meet the minimum criteria?  After all, some level of any model measure is defined by how much data you train it on.  It could be the case that we are right on the edge of having enough labeled data to train the model for all the imperative cases, but not enough to really ensure 90% percision, recall or some other meeasure.  If that is the case, then we could simply look to see if any of the folds does significantly worse than some notion of centrality, which could be a red flag on its own.  

Here we can set some deviance from the center for precision, recall or f1 score.  If a given fold falls below some deviance from centrality then we believe some intervention needs to be taken.  Let's look at an example::

	from sklearn import tree
	import pandas as pd
	import numpy as np
	import joblib

	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    if a + b + c > 11:
	        target = 1
	    else:
	        target = 0
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	clf = tree.DecisionTreeClassifier()
	X = df[["A", "B", "C"]]
	clf.fit(X, df["target"])
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")


Let's see a test::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_cv_precision_anomaly_detection():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    precision_tolerance = 0.2
	    test_suite.cross_val_precision_anomaly_detection(
	    	precision_tolerance, method='mean'
	    )

Here instead of setting an expectation of the precision, we set an expectation of the deviance from average precision.  So if the average is 0.7 and one of the folds scores is less than 5.0 then the test fails.  So it's important to have some lower boundary in place as well.  However we can be less stringent if we include this test.  A more complete test suite would likely be something like this::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_cv_precision_anomaly_detection():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    precision_tolerance = 0.2
	    test_suite.cross_val_precision_anomaly_detection(
	    	precision_tolerance, method='mean'
	    )

	def test_cv_precision_lower_boundary():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    min_averange = 0.7
	    test_suite.cross_val_precision_avg(
	    	min_average, method='mean'
	    )

Now we can say for sure, the precision should be at least 0.7 on average but can fall below up to 0.2 of that before we raise an error.  So 

Classifier Test Example - Cross Validation Anamoly Detection With Spread
========================================================================

In the previous example, we looked for a specific deviance now we'll make use of some properties of statistics to define what exactly we mean by an anamolous fold.  In order to do this, we'll look at deviance with respect to spread.  To make this concrete, let's walk through what that means::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_cv_precision_anomaly_detection():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    clf = joblib.load("model.joblib")

	    test_suite = ClassificationTests(clf, 
	    df, target_name, column_names)
	    precision_tolerance = 0.2
	    test_suite.cross_val_precision_anomaly_detection(
	    	precision_tolerance, method='mean'
	    )

Before we go through what's happening let's recall what cross validation is. The basic notion of cross validation is random samples are taken, called folds of from the training set, trains the algorithm with that data and tests against all the other folds.  For this reason, it is necessary that you have enough data such that you can learn a pattern from the data.  For more information on k-fold check out this article: https://machinelearningmastery.com/k-fold-cross-validation/.  

As you can see we require a precision tolerance of 0.2 per fold of the cross validation.  To understand how this comes into play, let's look at how cross validation anomaly detection is done generally in the library::

1. decide on the measure of center to use
2. calculate the average of all the scores (each score comes from a fold)
3. compute the list of deviances from the average
4. determine if the deviance from the average is every greater than the tolerance

So basically, this is a test for consistency on different folds of the data.  If the model performances above or below the tolerance bound on any of the folds, then the test fails.  This is really good if you need your model to act in an expected way, a lot of the time.  