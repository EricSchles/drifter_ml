# ML Testing

The goal of this module is to create a flexible and easy to use module for testing machine learning models, specifically those in scikit-learn.  

The tests will be readable enough that anyone can extend them to other frameworks and APIs with the major notions kept the same, but more or less the ideas will be extended, no work will be taken in this library to extend passed the scikit-learn API.

## Tests Covered

 * Testing Against Metrics
 	* Classification Tests
 		* precision lower boundary
 		* recall lower boundary
 		* f1 score lower boundary
 	* Regression Tests
 		* Mean Squared Error upper boundary
 		* Mean Absolute Error upper boundary
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
 	* correlation testing - this is about ensuring that the correlation for a given column
 							with previous data collected in the past does not change very much.
 							If the data does change then the model should possibly be rerun.
    * shape testing - this is about ensuring the general shape of for the given column does not
    				  change much over time.  The idea here is the same as the correlation tests.

## Future Features

* cross validation score testing
* add custom loss function
* add custom accuracy function
* add these tests: https://www.datasciencecentral.com/profiles/blogs/a-plethora-of-original-underused-statistical-tests
