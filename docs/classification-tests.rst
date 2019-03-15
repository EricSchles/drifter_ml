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
	    assert test_suite.classifier_testing(
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
	    performance_boundary = []
	    for size in range(1, 100000, 100):
	    	performance_boundary.append({
	    		"sample_size": size,
	    		"max_run_time": 10.0 # seconds
	    	})
	    assert test_suite.run_time_stress_test(
	        performance_boundary
	    )

This test ensures that from 1 to 100000 elements, the model never takes longer than 10 seconds.  

