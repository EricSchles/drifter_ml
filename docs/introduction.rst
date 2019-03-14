############
Introduction
############

Welcome to Drifter, a tool to help you test your machine learning models.  This testing framework is broken out semantically, so you can test different aspects of your machine learning system.  

The tests come in two general flavors, component tests, like this one that tests for a minimum precision per class::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_precision():
		clf = joblib.load("random_forest.joblib")
		test_data = pd.read_csv("test.csv")
		columns = test_data.columns.tolist()
		columns.remove("target")
		clf_tests = ClassificationTests(clf, test_data, "target", columns)
		classes = set(test_data["target"])
		precision_per_class = {klass: 0.9 for klass in classes}
		clf_tests.precision_lower_boundary_per_class(precision_per_class)


And an entire test suite that tests for precision, recall and f1 score in one test::

	from drifter_ml.classification_tests import ClassificationTests
	import joblib
	import pandas as pd

	def test_precision():
		clf = joblib.load("random_forest.joblib")
		test_data = pd.read_csv("test.csv")
		columns = test_data.columns.tolist()
		columns.remove("target")
		clf_tests = ClassificationTests(clf, test_data, "target", columns)
		classes = set(test_data["target"])
		precision_per_class = {klass: 0.9 for klass in classes}
		recall_per_class = {klass: 0.9 for klass in classes}
		f1_per_class = {klass: 0.9 for klass in classes}
		clf_tests.classifier_testing(
		precision_per_class,
		recall_per_class,
		f1_per_class
		)


The expectation at present is that all models follow the scikit learn api, which means there is an expectation of a `fit` and `predict` on all models.  This may appear exclusionary, but you can infact wrap keras models with scikit-learn style objects, allowing for the same api::

	from keras.models import Sequential
	from keras.layers import Dense
	from keras.wrappers.scikit_learn import KerasClassifier
	from sklearn.model_selection import StratifiedKFold
	from sklearn.model_selection import cross_val_score
	import numpy
	 
	# Function to create model, required for KerasClassifier
	def create_model():
		# create model
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model
	 
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	# load pima indians dataset
	dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
	# split into input (X) and output (Y) variables
	X = dataset[:,0:8]
	Y = dataset[:,8]
	# create model
	model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
	# evaluate using 10-fold cross validation
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(model, X, Y, cv=kfold)
	print(results.mean())

This means that traditional machine learning and deep learning are available for testing out of the box!