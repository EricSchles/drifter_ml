# ML Testing

The goal of this module is to create a flexible and easy to use module for testing machine learning models, specifically those in scikit-learn.  

The tests will be readable enough that anyone can extend them to other frameworks and APIs with the major notions kept the same, but more or less the ideas will be extended, no work will be taken in this library to extend passed the scikit-learn API.

You can [read the docs](https://drifter-ml.readthedocs.io/en/latest/) for a more detailed explaination.

[![Documentation Status](https://readthedocs.org/projects/drifter-ml/badge/?version=latest)](https://drifter-ml.readthedocs.io/en/latest/?badge=latest)

[![CircleCI](https://circleci.com/gh/EricSchles/drifter_ml.svg?style=svg)](https://circleci.com/gh/EricSchles/drifter_ml)
## Tests Covered

 * Testing Against Metrics
 	* Classification Tests
 		* Rule Based Testing:
	 		* precision lower boundary
	 		* recall lower boundary
	 		* f1 score lower boundary
	 		* AUC lower boundary
	 		* precision lower boundary per class
	 		* recall lower boundary per class
	 		* f1 score lower boundary per class
	 		* AUC lower boundary per class
 		* Decision Based Testing:
 			* precision fold below average
 			* recall fold below average
 			* f1 fold below average
 			* AUC fold below average
 			* precision fold below average per class
 			* recall fold below average per class
 			* f1 fold below average per class
 			* AUC fold below average per class
 		* Against New Predictions
 			* proportion of predictions per class
 			* class imbalance tests
 			* probability distribution similarity tests
 			* calibration tests
 	* Regression Tests
 		* Rule Based Testing:
 		 	* Mean Squared Error upper boundary
 			* Median Absolute Error upper boundary
 		* Decision Based Testing:
 			* Mean Squared Error fold above average
 			* Median Absolute Error fold above average
 * Testing Against Run Time Performance
 	* prediction run time for simulated samples of size X
 * Testing Against Input Data
  	* percentage of correct imputes for any columns requiring imputation
 	* dataset testing - http://www.vldb.org/pvldb/vol11/p1781-schelter.pdf 
 * Memoryful Tests
 	* cluster testing - this is about the overall structure of the data
 		If the number of clusters increases or decreases substantially that 
 		should be an indicator that the data has changed enough that things
 		should possibly be rerun
 	* correlation testing - this is about ensuring that the correlation for a given column with previous data collected in the past does not change very much. If the data does change then the model should possibly be rerun.
    * shape testing - this is about ensuring the general shape of for the given column does not change much over time.  The idea here is the same as the correlation tests.

## Possible Issues

Some known issues with this, any machine learning tests are going to require human interaction because of type 1 and type 2 error for statistical tests.  Additionally, one simply needs to interrogate models from a lot of angles.  It can't be from just one angle.  So I'm not even sure if the overall notion of adding ML testing into a CI pipeline is really feasible.

## Future Features

* cross validation score testing
* add custom loss function
* add custom accuracy function
* add these tests: https://www.datasciencecentral.com/profiles/blogs/a-plethora-of-original-underused-statistical-tests
* clustering for classification

## References

* https://dzone.com/articles/quality-assurancetesting-the-machine-learning-mode
* https://medium.com/datadriveninvestor/how-to-perform-quality-assurance-for-ml-models-cef77bbbcfb
* Explaination of UAT: https://www.techopedia.com/definition/3887/user-acceptance-testing-uat
* https://mice.cs.columbia.edu/getTechreport.php?techreportID=419&format=pdf
* https://www.xenonstack.com/blog/unit-testing-tdd-bdd-deep-machine-learning/