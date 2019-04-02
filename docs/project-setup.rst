#############
Project Setup
#############

Regression and Classification Tests
===================================

If you are going to use regression or classification tests, you'll need to do a bit of setup.  The first step is making sure you have a test set with labeled data that you can trust. It is recommended that you break your initial labeled dataset up into test and train and keep the test for both the model generation phase as well as for model monitoring throughout.

A good rule of thumb is to have 70% train, and 30% test.  Other splits may be ideal, depending on the needs of your project.  You can setup test and train using existing tools from sklearn as follows::

	 from sklearn.model_selection import train_test_split
	 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Once you have your two datasets you can train your model with the training set, as is typical::

	from sklearn import tree
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	import joblib

	df = pd.DataFrame()
	for _ in range(5000):
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
	y = df["target"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	clf.fit(X_train, y_train)
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")
	test_data = pd.DataFrame()
	test_data[["A", "B", "C"]]
	test_data["target"] = y_test
	test_data.to_csv("test_data.csv")

Then you can test against your model before you put it into production as follows::

	import joblib
	import pandas as pd
	from sklearn.metrics import f1_score

	clf = joblib.load("model.joblib")
	test_data = pd.read_csv("test_data.csv")
	y_pred = clf.predict(test_data[["A", "B", "C"]])
	y_true = test_data["target"]
	print(f1_score(y_true, y_pred))

It's worth noting that one score is likely never good enough, you need to include multiple measures to ensure your model is not simply fitting towards a single measure.  Assuming the measures are good enough you can move onto productionizing your model.

Strategies For Testing Your Productionized Model
================================================

Once you've put your model into production there are a few strategies for making sure your model continues to meet your requirements:

1. Using the test set from training - Gathering predictions and data that is new and then training a new classifier or regressor with the new data and new predictions.  Then test against the test set you've set aside.  If the measures stay approximately the same, it's possible your model is performing as expected.  It's important that the new classifier have the same hyper parameters as the one in production as well as using the same versions for all associated code that creates the new model object.

2. Generating a new test set from a process -Gathering new data and new predictions from the production model and then training a new classifier or regressor with the new data and new predictions.  Then manually label the same set of new data, either via some people process or other process you believe to be able to generate faithful labels.  Then carry out the same validation as in strategy one.

3. Generating a new test set from a process and then do label propagation - Gathering new data and new predictions from the production model and then training a new classifier or regressor with the new data and new predictions.  Then manually label a small set of the new data in some manor.  Then generate a new set of labels via label propagation and finally use this newly created test set against your newly trained classifier.


Using The Test Set From Training
================================

So the above description is a bit terse so let's break it down with some example code to inform your own project setup.  First let's assume that you have some data to train on and test on::

	from sklearn import tree
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	import joblib

	df = pd.DataFrame()
	for _ in range(5000):
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
	y = df["target"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	clf.fit(X_train, y_train)
	joblib.dump(clf, "model.joblib")
	df.to_csv("data.csv")
	test_data = pd.DataFrame()
	test_data[["A", "B", "C"]]
	test_data["target"] = y_test
	test_data.to_csv("test_data.csv")

Next we need to test our model to make sure it's performing well enough to go into production::

	import joblib
	import pandas as pd
	from sklearn.metrics import classification_report

	clf = joblib.load("model.joblib")
	test_data = pd.read_csv("test_data.csv")
	y_pred = clf.predict(test_data[["A", "B", "C"]])
	y_true = test_data["target"]
	print(classification_report(y_true, y_pred))

Let's assume everything met our minimum criteria for going to production. Now we are ready to put our model into production!! For this we'll need to write our test such that it makes use of the test data, our new data and our new predictions.  For the purposes of the below example, assume you've been saving new data and new predictions to a csv called new_data.csv, that you have saved your production model in a file called model.joblib and that you have test data saved to test_data.csv.  Below is an example test you might write using the framework::

	import joblib
	import pandas as pd
	from sklearn import tree
	from drifter_ml import classification_tests

	def generate_model_from_production_data():
		new_data = pd.read_csv("new_data.csv")
		prod_clf = joblib.load("model.joblib")
		test_data = pd.read_csv("test_data.csv")
		return test_data, new_data, prod_clf

	def test_precision():
		test_data, new_data, prod_clf = generate_model_from_production_data()
		column_names = ["A", "B", "C"]
	    target_name = "target"
	    test_clf = tree.DecisionTreeClassifier()
	    test_clf.set_params(**prod_clf.get_params())
	    X = new_data[column_names]
	    y = new_data[target_name]
	    test_clf.fit(X, y)

	    test_suite = ClassificationTests(test_clf, 
	    	test_data, target_name, column_names)
	    classes = list(df.target.unique())
	    lower_bound_requirement = {klass: 0.9 for klass in classes}
	    assert test_suite.precision_lower_boundary_per_class(
	        lower_bound_requirement
	    )

Notice that we train on the production data and labels (in this case in target) and then test against the labels we know.  Here we use the lower_bound_requirement variable to set the expectation for how well the model should do against the test set.  If the labels generated by the production model train a model that performs as well on the test data as the production model did on the test set, then we have some confidence in the labels it produces.  This is probably not the only way one could do this comparison, if you come up with something better, please share back out to the project!