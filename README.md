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
 	* testing consistency of columns
 	* percentage of correct imputes for any columns requiring imputation
 

