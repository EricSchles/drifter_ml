#################
Regression Tests
#################

So this section will likely be the most confusing for anyone coming from classical software engineering.  Here regression refers to a model that outputs a floating point number, instead of a class.  The biggest important difference between classification and regression is, the numbers produced by regression are "real" numbers.  So they actually have magnitude, direction, a sense of scale, etc.  

Classification returns a "class".  Which means class "1" has no ordering relationship with class "2".  So you shouldn't compare these with ordering.

In any event, the regression tests break out into the follow categories:

1. Establish a baseline maximum error tolerance based on a model measure
2. Establish a tolerance level for deviance from the average fold error
3. Stress testing for the speed of calculating new values
4. Comparison of the current model against new models for the above defined measures
5. Comparison of the speed of performance against new models

Upper Bound Regression Metrics
==============================

Each of the following examples ensures that your model meets a minimum criteria, which should be decided based on the need of your use-case.  One simple way to do this is to define failure by how many dollars it will cost you for every unit amount your model is off on average.  

Mean Squared Error and Median Absolute Error are great tools for ensuring your regressor optimizes for least error.  The scale of that error will be entirely context specific.

That is why they are basis of the set of tests found below.

Regression Test Example - Model Metrics
=======================================

Suppose you had the following model::

	from sklearn import linear_model
	import pandas as pd
	import numpy as np
	import joblib

	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    target = 5*a + 3*b + c
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	reg = linear_model.LinearRegression()
	X = df[["A", "B", "C"]]
	reg.fit(X, df["target"])
	joblib.dump(reg, "model.joblib")
	df.to_csv("data.csv")

We could write the following set of tests to ensure this model does well::

	from drifter_ml.regression_tests import RegressionTests
	import joblib
	import pandas as pd

	def test_mse():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    reg = joblib.load("model.joblib")

	    test_suite = RegressionTests(reg, 
	    df, target_name, column_names)
	    mse_boundary = 15
	    assert test_suite.mse_upper_boundary(mse_boundary)

	def test_mae():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    reg = joblib.load("model.joblib")

	    test_suite = RegressionTests(reg, 
	    df, target_name, column_names)
	    mae_boundary = 10
	    assert test_suite.mae_upper_boundary(mae_boundary)

Or you could simply write one test for all three::

	from drifter_ml.regression_tests import RegressionTests
	import joblib
	import pandas as pd

	def test_mse_mae():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    reg = joblib.load("model.joblib")

	    test_suite = RegressionTests(reg, 
	    df, target_name, column_names)
	    mse_boundary = 15
	    mae_boundary = 10
	    assert test_suite.regression_testing(mse_boundary,
	    					 mae_boundary)

Regression Test Example - Model Speed
=====================================

Additionally, you can test to ensure your regressor performs, even under load.  Assume we have the same model as before::

	from sklearn import linear_model
	import pandas as pd
	import numpy as np
	import joblib

	df = pd.DataFrame()
	for _ in range(1000):
	    a = np.random.normal(0, 1)
	    b = np.random.normal(0, 3)
	    c = np.random.normal(12, 4)
	    target = 5*a + 3*b + c
	    df = df.append({
	        "A": a,
	        "B": b,
	        "C": c,
	        "target": target
	    }, ignore_index=True)

	reg = linear_model.LinearRegression()
	X = df[["A", "B", "C"]]
	reg.fit(X, df["target"])
	joblib.dump(reg, "model.joblib")
	df.to_csv("data.csv")

Now we test to ensure the model predicts new labels within our constraints::

	from drifter_ml.regression_tests import RegressionTests
	import joblib
	import pandas as pd

	def test_mse_mae_speed():
	    df = pd.read_csv("data.csv")
	    column_names = ["A", "B", "C"]
	    target_name = "target"
	    reg = joblib.load("model.joblib")

	    test_suite = RegressionTests(reg, 
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

