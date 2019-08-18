from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_validate, cross_val_predict
from functools import partial
from sklearn.model_selection import KFold
from sklearn.base import clone

class FixedClassificationMetrics():
    def __init__(self):
        pass
    
    def precision_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        """
        The Scikit-Learn precision score, see the full documentation here:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        
        The difference between this precision score and the one in scikit-learn,
        is we fix a small bug.  When all the values in y_true are zero and
        y_pred are zero the precision_score returns one. (Which Scikit-learn
        does not do at present).
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
          Ground truth (correct) target values.
        y_pred : 1d array-like, or label indicator array / sparse matrix
          Estimated targets as returned by a classifier
        labels : list, optional
          The set of labels to include when average != binary, and their order
          if average is None.  Labels present in the data can be excluded, for
          example to calculate a multiclass average ignoring a majority negative
          class, while labels not present in the data will result in 0 components
          in a macro average. For multilabel targets, labels are column indices.
          By default, all labels in y_true and y_pred are used in sorted order.
        pos_label : str or int, 1 by default
          The class to report if average='binary' and the data is binary.  If
          the data are multiclass or multilabel, this will be ignored; setting
          labels=[pos_label] and average != 'binary' will report scores for
          that label only.
        average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary' : string
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro' : string
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro' : string
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted' : string
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples' : string
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        sample_weight: array-like
          array-like of shape = [n_samples], optional
             Sample weights.

        Returns
        -------
         precision: float 
           (if average is not None) or array of float, shape = [n_unique_labels]
           Precision of the positive class in binary classification or weighted
           average of the precision of each class for the multiclass task.        
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.precision_score(y_true,
                                           y_pred,
                                           labels=labels,
                                           pos_label=pos_label,
                                           average=average,
                                           sample_weight=sample_weight)

    def recall_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        """
        The Scikit-Learn precision score, see the full documentation here:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        
        The difference between this recall score and the one in scikit-learn,
        is we fix a small bug.  When all the values in y_true are zero and
        y_pred are zero the recall_score returns one. (Which Scikit-learn
        does not do at present).
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
          Ground truth (correct) target values.
        y_pred : 1d array-like, or label indicator array / sparse matrix
          Estimated targets as returned by a classifier
        labels : list, optional
          The set of labels to include when average != binary, and their order
          if average is None.  Labels present in the data can be excluded, for
          example to calculate a multiclass average ignoring a majority negative
          class, while labels not present in the data will result in 0 components
          in a macro average. For multilabel targets, labels are column indices.
          By default, all labels in y_true and y_pred are used in sorted order.
        pos_label : str or int, 1 by default
          The class to report if average='binary' and the data is binary.  If
          the data are multiclass or multilabel, this will be ignored; setting
          labels=[pos_label] and average != 'binary' will report scores for
          that label only.
        average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        sample_weight : array-like
           array-like of shape = [n_samples], optional
           Sample weights.

        Returns
        -------
        recall : float 
           (if average is not None) or array of float, shape = [n_unique_labels]
           Recall of the positive class in binary classification or weighted
           average of the recall of each class for the multiclass task.        
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.recall_score(y_true,
                                        y_pred,
                                        labels=labels,
                                        pos_label=pos_label,
                                        average=average,
                                        sample_weight=sample_weight)

    def f1_score(self, y_true, y_pred,
                        labels=None, pos_label=1, average='binary', sample_weight=None):
        """
        The Scikit-Learn precision score, see the full documentation here:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        
        The difference between this f1 score and the one in scikit-learn,
        is we fix a small bug.  When all the values in y_true are zero and
        y_pred are zero the f1_score returns one. (Which Scikit-learn
        does not do at present).
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
          Ground truth (correct) target values.
        y_pred : 1d array-like, or label indicator array / sparse matrix
          Estimated targets as returned by a classifier
        labels : list, optional
          The set of labels to include when average != binary, and their order
          if average is None.  Labels present in the data can be excluded, for
          example to calculate a multiclass average ignoring a majority negative
          class, while labels not present in the data will result in 0 components
          in a macro average. For multilabel targets, labels are column indices.
          By default, all labels in y_true and y_pred are used in sorted order.
        * pos_label : str or int, 1 by default
          The class to report if average='binary' and the data is binary.  If
          the data are multiclass or multilabel, this will be ignored; setting
          labels=[pos_label] and average != 'binary' will report scores for
          that label only.
        * average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        * sample_weight : array-like
           array-like of shape = [n_samples], optional. Sample weights.

        Returns
        -------
         * f1: float 
           (if average is not None) or array of float, shape = [n_unique_labels]
           F1 score of the positive class in binary classification or weighted
           average of the f1 scores of each class for the multiclass task.        
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            return metrics.f1_score(y_true,
                                    y_pred,
                                    labels=labels,
                                    pos_label=pos_label,
                                    average=average,
                                    sample_weight=sample_weight)

    def _prepare_for_per_class_comparison(self, y_true, y_pred):
        # this means the per class roc curve can never be zero
        # but allows us to do per class classification
        if (np.unique(y_true) == 0).all():
            y_true = np.append(y_true, [1])
            y_pred = np.append(y_pred, [1])
        if (np.unique(y_true) == 1).all():
            y_true = np.append(y_true, [0])
            y_pred = np.append(y_pred, [0])
        return y_true, y_pred
                
    def roc_auc_score(self, y_true, y_pred,
                      labels=None, pos_label=1, average='micro', sample_weight=None):
        """
        The Scikit-Learn precision score, see the full documentation here:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        
        The difference between this roc_auc score and the one in scikit-learn,
        is we fix a small bug.  When all the values in y_true are zero and
        y_pred are zero the roc_auc_score returns one. (Which Scikit-learn
        does not do at present).
        
        Parameters
        ----------
        * y_true : 1d array-like, or label indicator array / sparse matrix
          Ground truth (correct) target values.
        * y_pred : 1d array-like, or label indicator array / sparse matrix
          Estimated targets as returned by a classifier
        * labels: list, optional
          The set of labels to include when average != binary, and their order
          if average is None.  Labels present in the data can be excluded, for
          example to calculate a multiclass average ignoring a majority negative
          class, while labels not present in the data will result in 0 components
          in a macro average. For multilabel targets, labels are column indices.
          By default, all labels in y_true and y_pred are used in sorted order.
        * pos_label : str or int, 1 by default
          The class to report if average='binary' and the data is binary.  If
          the data are multiclass or multilabel, this will be ignored; setting
          labels=[pos_label] and average != 'binary' will report scores for
          that label only.
        * average : string
          string, [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        * sample_weight: array-like
          array-like of shape = [n_samples], optional Sample weights.
        Returns
        -------
         * roc_auc: float 
           (if average is not None) or array of float, shape = [n_unique_labels]
           Roc_auc score of the positive class in binary classification or weighted
           average of the roc_auc scores of each class for the multiclass task.        
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (y_true == y_pred).all() == True:
            return 1.0
        else:
            y_true, y_pred = self._prepare_for_per_class_comparison(y_true, y_pred)
            return metrics.roc_auc_score(y_true,
                                         y_pred,
                                         average=average,
                                         sample_weight=sample_weight)

class ClassificationTests(FixedClassificationMetrics):
    """
    The general goal of this class it to test classification 
    algorithms.  
    The tests in this class move from simple to sophisticated:

    * cross_val_average : the average of all folds must be above some number
    * cross_val_lower_boundary : each fold must be above the lower boundary
    * lower_boundary_per_class : each class must be above a given lower boundary the lower boundary per class can be different
    * cross_val_anomaly_detection : the score for each fold must have a deviance from the average below a set tolerance
    * cross_val_per_class_anomaly_detection : the score for each class for each fold must have a deviance from the average below a set tolerance
    
    As you can see, at each level of sophistication we need more data to get
    representative sets.  But if more data is available, then we are able
    to test increasingly more cases.  The more data we have to test against,
    the more sure we can be about how well our model does.  
    
    Another lense to view each classes of tests, is with respect to stringency.
    If we need our model to absolutely work all the time, it might be important
    to use the most sophisticated class - something with cross validation, per class.
    It's worth noting, that increased stringency isn't always a good thing.
    Statistical models, by definition aren't supposed to cover every case perfectly.
    They are supposed to be flexible.  So you should only use the most strigent
    checks if you truly have a ton of data.  Otherwise, you will more or less
    'overfit' your test suite to try and look for errors.  Testing in machine learning
    like in software engineering is very much an art.  You need to be sure to cover
    enough cases, without going overboard.
    """
    def __init__(self,
                 clf,
                 test_data,
                 target_name,
                 column_names):
        self.clf = clf
        self.test_data = test_data
        self.column_names = column_names
        self.target_name = target_name
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)
            
    def get_test_score(self, cross_val_dict):
        return list(cross_val_dict["test_score"])
    
    # add cross validation per class tests
    def precision_cv(self, cv, average='binary'):
        """
        This method performs cross-validation over precision.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform
        * average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        Returns
        -------
        Returns a scores of the k-fold precision.
        """
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        precision = metrics.make_scorer(precision_score)
        result =  cross_validate(self.clf, self.X,
                                 self.y, cv=cv,
                                 scoring=(precision))
        return self.get_test_score(result)
    
    def recall_cv(self, cv, average='binary'):
        """
        This method performs cross-validation over recall.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform
        * average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).
        Returns
        -------
        Returns a scores of the k-fold recall.
        """
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        recall = metrics.make_scorer(recall_score)
        result = cross_validate(self.clf, self.X,
                                self.y, cv=cv,
                                scoring=(recall))
        return self.get_test_score(result)
    
    def f1_cv(self, cv, average='binary'):
        """
        This method performs cross-validation over f1-score.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform
        * average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).

        Returns
        -------
        Returns a scores of the k-fold f1-score.
        """
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        f1 = metrics.make_scorer(f1_score)
        result = cross_validate(self.clf, self.X,
                                self.y, cv=cv,
                                scoring=(f1))
        return self.get_test_score(result)

    def roc_auc_cv(self, cv, average="micro"):
        """
        This method performs cross-validation over roc_auc.
        
        Parameters
        ----------
        * cv : integer
          The number of cross validation folds to perform
        * average : string, 
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.
          
          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives, 
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean. 
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters 
             'macro' to account for label imbalance; it can result in an F-score 
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from 
             accuracy_score).

        Returns
        -------
        Returns a scores of the k-fold roc_auc.
        """
        roc_auc_score = partial(self.roc_auc_score, average=average)
        roc_auc = metrics.make_scorer(roc_auc_score)
        result = cross_validate(self.clf, self.X,
                                self.y, cv=cv,
                                scoring=(roc_auc))
        return self.get_test_score(result)
    
    def _cross_val_avg(self, scores, minimum_center_tolerance, method='mean'):
        avg, _ = self.describe_scores(scores, method)
        if avg < minimum_center_tolerance:
            return False
        return True

    def _get_per_class(self, y_true, y_pred, metric):
        class_measures = {klass: None for klass in self.classes}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, y_true[y_true == klass].index, axis=0)
            y_class = y_true[y_true == klass]
            class_measures[klass] = metric(y_class, y_pred_class)
        return class_measures

    def _per_class_cross_val(self, metric, cv, random_state=42):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        clf = clone(self.clf)
        scores = []
        for train, test in kfold.split(self.test_data):
            train_data = self.test_data.loc[train]
            test_data = self.test_data.loc[test]
            clf.fit(train_data[self.column_names], train_data[self.target_name])
            y_pred = clf.predict(test_data[self.column_names])
            y_true = test_data[self.target_name]
            y_true.index = list(range(len(y_true)))
            scores.append(self._get_per_class(y_true, y_pred, metric))
        return scores

    def _cross_val_anomaly_detection(self, scores, tolerance, method='mean'):
        avg, _ = self.describe_scores(scores, method)
        deviance_from_avg = [abs(score - avg)
                             for score in scores]
        for deviance in deviance_from_avg:
            if deviance > tolerance:
                return False
        return True

    def _cross_val_per_class_anomaly_detection(self, metric,
                                               tolerance, cv, method='mean'):
        scores_per_fold = self._per_class_cross_val(metric, cv)
        results = [] 
        for klass in self.classes:
            scores = [score[klass] for score in scores_per_fold]
            results.append(
                self._cross_val_anomaly_detection(
                    scores, tolerance, method=method
                )
            )
        return all(results)

    def _cross_val_lower_boundary(self, scores, lower_boundary):
        for score in scores:
            if score < lower_boundary:
                return False
        return True

    def _anomaly_detection(self, scores, tolerance, method):
        center, spread = self.describe_scores(scores, method)
        for score in scores:
            if score < center - (spread * tolerance):
                return False
        return True

    def _per_class(self, y_pred, metric, lower_boundary):
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            if metric(y_class, y_pred_class) < lower_boundary[klass]:
                return False
        return True

    def is_binary(self):
        """
        If number of classes == 2 returns True
        False otherwise
        """
        num_classes = len(set(self.classes))
        if num_classes == 2:
            return True
        return False
    
    def roc_auc_exception(self):
        """
        Ensures roc_auc score is used correctly.
        ROC AUC is only defined for binary classification.
        """
        if not self.is_binary():
            raise Exception("roc_auc is only defined for binary classifiers")

    def reset_average(self, average):
        """
        Resets the average to the correct thing.
        If the number of classes are not binary,
        Then average is changed to micro.
        Otherwise, return the current average.
        """
        if not self.is_binary() and average == 'binary':
            return 'micro'
        return average

    def cross_val_per_class_precision_anomaly_detection(self, tolerance: float,
                                                        cv=3, average='binary',
                                                        method='mean'):
        """
        This checks the cross validated per class percision score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average precision
        cv : int
          the number of folds to consider
        average : string
          how to calculate the precision
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for precision
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for precision
        """
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        return self._cross_val_per_class_anomaly_detection(
            precision_score, tolerance, cv, method=method
        )

    def cross_val_per_class_recall_anomaly_detection(self, tolerance: float,
                                                     cv=3, average='binary',
                                                     method='mean'):
        """
        This checks the cross validated per class recall score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average recall
        cv : int
          the number of folds to consider
        average : string
          how to calculate the recall
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for recall
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for recall
        """
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        return self._cross_val_per_class_anomaly_detection(
            recall_score, tolerance, cv, method=method)

    def cross_val_per_class_f1_anomaly_detection(self, tolerance: float,
                                                 cv=3, average='binary',
                                                 method='mean'):
        """
        This checks the cross validated per class f1 score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average f1 score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the f1 score
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for f1 score
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for f1 score
        """
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        return self._cross_val_per_class_anomaly_detection(
            f1_score, tolerance, cv, method=method
        )

    def cross_val_per_class_roc_auc_anomaly_detection(self, tolerance: float,
                                                      cv=3, average="micro",
                                                      method='mean'):
        """
        This checks the cross validated per class roc auc score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average roc auc
        cv : int
          the number of folds to consider
        average : string
          how to calculate the roc auc
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for roc auc
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for roc auc
        """
        self.roc_auc_exception()
        roc_auc_score = partial(self.roc_auc_score, average=average)
        return self._cross_val_per_class_anomaly_detection(
            roc_auc_score, tolerance, cv, method=method
        )
    
    def cross_val_precision_anomaly_detection(self, tolerance: float,
                                              cv=3, average='binary',
                                              method='mean'):
        """
        This checks the k fold (cross validation) precision score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average precision
        cv : int
          the number of folds to consider
        average : string
          how to calculate the precision
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for precision
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for precision
        """
        average = self.reset_average(average)
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_anomaly_detection(
            scores, tolerance, method=method
        )
    
    def cross_val_recall_anomaly_detection(self, tolerance: float,
                                           cv=3, average='binary', method='mean'):
        """
        This checks the k fold (cross validation) recall score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average recall
        cv : int
          the number of folds to consider
        average : string
          how to calculate the recall
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for recall
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for recall
        """
        average = self.reset_average(average)
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_anomaly_detection(
            scores, tolerance, method=method
        )
    
    def cross_val_f1_anomaly_detection(self, tolerance: float,
                                       cv=3, average='binary', method='mean'):
        """
        This checks the k fold (cross validation) f1 score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average f1 score
        cv : int 
          the number of folds to consider
        average : string
          how to calculate the f1 score
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for f1 score
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for f1 score
        """
        average = self.reset_average(average)
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_anomaly_detection(
            scores, tolerance, method=method
        )

    def cross_val_roc_auc_anomaly_detection(self, tolerance: float,
                                            cv=3, average="micro", method='mean'):
        """
        This checks the k fold (cross validation) roc auc score, based on 
        anolamies.  The way the anomaly detection scheme works is, an 
        average is calculated and then if the deviance from the average is 
        greater than the set tolerance, then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance from the average roc auc
        cv : int
          the number of folds to consider
        average : string
          how to calculate the roc auc
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the deviances from average for all the folds 
        are above tolerance for roc auc
        False if any of the deviances from the average for any of 
        the folds are below the tolerance for roc auc
        """
        self.roc_auc_exception()
        scores = self.roc_auc_cv(cv, average=average)
        return self._cross_val_anomaly_detection(
            scores, tolerance, method=method
        )
        
    def cross_val_precision_avg(self, minimum_center_tolerance,
                                cv=3, average='binary', method='mean'):
        """
        This generates the k fold (cross validation) precision scores, 
        then based on computes the average of those scores.  
        The way the average scheme works is, an average is 
        calculated and then if the average is less 
        than the minimum tolerance, then False is returned.
        
        Parameters
        ----------
        minimum_center_tolerance : float
          the average precision must be greater than this number
        cv : int
          the number of folds to consider
        average : string 
          how to calculate the precision
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the folds of the precision are greater than
        the minimum_center_tolerance
        False if the average folds for the precision are less than
        the minimum_center_tolerance
        """
        average = self.reset_average(average)
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_avg(
            scores, minimum_center_tolerance, method=method)

    def cross_val_recall_avg(self, minimum_center_tolerance,
                             cv=3, average='binary', method='mean'):
        """
        This generates the k fold (cross validation) recall scores, 
        then based on computes the average of those scores.  
        The way the average scheme works is, an average is 
        calculated and then if the average is less 
        than the minimum tolerance, then False is returned.
        
        Parameters
        ----------
        minimum_center_tolerance : float
          the average recall must be greater than this number
        cv : int
          the number of folds to consider
        average : string
          how to calculate the recall
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the folds of the recall are greater than
        the minimum_center_tolerance
        False if the average folds for the recall are less than
        the minimum_center_tolerance
        """

        average = self.reset_average(average)
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_avg(
            scores, minimum_center_tolerance, method=method)

    def cross_val_f1_avg(self, minimum_center_tolerance,
                         cv=3, average='binary', method='mean'):
        """
        This generates the k fold (cross validation) f1 scores, 
        then based on computes the average of those scores.  
        The way the average scheme works is, an average is 
        calculated and then if the average is less 
        than the minimum tolerance, then False is returned.
        
        Parameters
        ----------
        minimum_center_tolerance : float
          the average f1 score must be greater than this number
        cv : int
          the number of folds to consider
        average : string
          how to calculate the f1 score
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the folds of the f1 score are greater than
        the minimum_center_tolerance
        False if the average folds for the f1 score are less than
        the minimum_center_tolerance
        """

        average = self.reset_average(average)
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_avg(
            scores, minimum_center_tolerance, method=method)

    def cross_val_roc_auc_avg(self, minimum_center_tolerance,
                              cv=3, average='micro', method='mean'):
        """
        This generates the k fold (cross validation) roc auc scores, 
        then based on computes the average of those scores.  
        The way the average scheme works is, an average is 
        calculated and then if the average is less 
        than the minimum tolerance, then False is returned.
        
        Parameters
        ----------
        minimum_center_tolerance : float
          the average roc auc must be greater than this number
        cv : int
          the number of folds to consider
        average : string
          how to calculate the roc auc
        method: string
          how to calculate the center
        
        Returns
        -------
        True if all the folds of the roc auc are greater than
        the minimum_center_tolerance
        False if the average folds for the roc auc are less than
        the minimum_center_tolerance
        """
        self.roc_auc_exception()
        scores = self.roc_auc_cv(cv, average=average)
        return self._cross_val_avg(
            scores, minimum_center_tolerance, method=method)
    
    def cross_val_precision_lower_boundary(self, lower_boundary,
                                           cv=3, average='binary'):
        """
        This is possibly the most naive stragey,
        it generates the k fold (cross validation) precision scores, 
        if any of the k folds are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : float
          the lower boundary for a given precision score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the folds of the precision scores are 
        greater than the lower_boundary
        False if the folds for the precision scores are 
        less than the lower_boundary
        """
        average = self.reset_average(average)
        scores = self.precision_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
        
    def cross_val_recall_lower_boundary(self, lower_boundary,
                                        cv=3, average='binary'):
        """
        This is possibly the most naive stragey,
        it generates the k fold (cross validation) recall scores, 
        if any of the k folds are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : float
          the lower boundary for a given recall score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the recall
        
        Returns
        -------
        True if all the folds of the recall scores are greater than
        the lower_boundary
        False if the folds for the recall scores are less than
        the lower_boundary
        """
        average = self.reset_average(average)
        scores = self.recall_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
        
    def cross_val_f1_lower_boundary(self, lower_boundary,
                                    cv=3, average='binary'):
        """
        This is possibly the most naive stragey,
        it generates the k fold (cross validation) f1 scores, 
        if any of the k folds are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : float
          the lower boundary for a given f1 score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the f1 score
        
        Returns
        -------
        True if all the folds of the f1 scores are greater than
        the lower_boundary
        False if the folds for the f1 scores are less than
        the lower_boundary
        """

        average = self.reset_average(average)
        scores = self.f1_cv(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)

    def cross_val_roc_auc_lower_boundary(self, lower_boundary,
                                         cv=3, average='micro'):
        """
        This is possibly the most naive stragey,
        it generates the k fold (cross validation) roc auc scores, 
        if any of the k folds are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : float
          the lower boundary for a given roc auc score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the roc auc
        
        Returns
        -------
        True if all the folds of the roc auc scores are greater than
        the lower_boundary
        False if the folds for the roc auc scores are less than
        the lower_boundary
        """

        self.roc_auc_exception()
        scores = self.roc_auc(cv, average=average)
        return self._cross_val_lower_boundary(scores, lower_boundary)
    
    def cross_val_classifier_testing(self,
                                     precision_lower_boundary: float,
                                     recall_lower_boundary: float,
                                     f1_lower_boundary: float,
                                     cv=3, average='binary'):
        """
        runs the cross validated lower boundary methods for:
        * precision, 
        * recall, 
        * f1 score
        The basic idea for these three methods is to check if
        the accuracy metric stays above a given lower bound.
        We can set the same precision, recall, or f1 score lower boundary
        or specify each depending on necessary criteria.
        
        Parameters
        ----------
        precision_lower_boundary : float
          the lower boundary for a given precision score
        recall_lower_boundary : float
          the lower boundary for a given recall score
        f1_lower_boundary : float
          the lower boundary for a given f1 score
        cv : int
          the number of folds to consider
        average : string
          how to calculate the metrics (precision, recall, f1)
        
        Returns
        -------
        Returns True if precision, recall and f1 tests
        work.  
        False otherwise
        """
        average = self.reset_average(average)
        precision_test = self.cross_val_precision_lower_boundary(
            precision_lower_boundary, cv=cv, average=average)
        recall_test = self.cross_val_recall_lower_boundary(
            recall_lower_boundary, cv=cv, average=average)
        f1_test = self.cross_val_f1_lower_boundary(
            f1_lower_boundary, cv=cv, average=average)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def trimean(self, data):
        """
        I'm exposing this as a public method because
        the trimean is not implemented in enough packages.
        
        Formula:
        (25th percentile + 2*50th percentile + 75th percentile)/4
        
        Parameters
        ----------
        data : array-like
          an iterable, either a list or a numpy array

        Returns
        -------
        the trimean: float
        """
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        median = np.median(data)
        return (q1 + 2*median + q3)/4

    def trimean_absolute_deviation(self, data):
        """
        The trimean absolute deviation is the
        the average distance from the trimean.
        
        Parameters
        ----------
        data : array-like
          an iterable, either a list or a numpy array

        Returns
        -------
        the average distance to the trimean: float
        """
        trimean = self.trimean(data)
        numerator = [abs(elem - trimean) for elem in data]
        return sum(numerator)/len(data)
        
    def describe_scores(self, scores, method):
        """
        Describes scores.
        
        Parameters
        ----------
        scores : array-like
          the scores from the model, as a list or numpy array
        method : string
          the method to use to calculate central tendency and spread
        
        Returns
        -------
        Returns the central tendency, and spread
        by method.
        
        Methods:
        mean:
        * central tendency: mean
        * spread: standard deviation
        
        median:
        * central tendency: median
        * spread: interquartile range
        
        trimean:
        * central tendency: trimean
        * spread: trimean absolute deviation
        """
        if method == "mean":
            return np.mean(scores), np.std(scores)
        elif method == "median":
            return np.median(scores), stats.iqr(scores)
        elif method == "trimean":
            return self.trimean(scores), self.trimean_absolute_deviation(scores)

    def spread_cross_val_precision_anomaly_detection(self, tolerance,
                                                     method="mean",
                                                     cv=10,
                                                     average='binary'):
        """
        This is a somewhat intelligent stragey,
        it generates the k fold (cross validation) precision scores, 
        if any of the k folds score less than the center - (spread * tolerance),
        then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance modifier for how far below the 
          center the score can be before a false is returned
        method : string
          see describe for more details.
          * mean : the center is the mean, the spread is standard
                   deviation.
          * median : the center is the median, the spread is
                     the interquartile range.
          * trimean : the center is the trimean, the spread is
                      trimean absolute deviation.
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the folds of the precision scores are greater than
        the center - (spread * tolerance)
        False if the folds for the precision scores are less than
        the center - (spread * tolerance)
        """
        average = self.reset_average(average)
        scores = self.precision_cv(cv, average=average)
        return self._anomaly_detection(scores, tolerance, method)
    
    def spread_cross_val_recall_anomaly_detection(self, tolerance,
                                                  method="mean",
                                                  cv=3,
                                                  average='binary'):
        """
        This is a somewhat intelligent stragey,
        it generates the k fold (cross validation) recall scores, 
        if any of the k folds score less than the center - (spread * tolerance),
        then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance modifier for how far below the 
          center the score can be before a false is returned
        method : string
          see describe for more details.
          * mean : the center is the mean, the spread is standard
                   deviation.
          * median : the center is the median, the spread is
                     the interquartile range.
          * trimean : the center is the trimean, the spread is
                      trimean absolute deviation.
        average : string
          how to calculate the recall
        
        Returns
        -------
        True if all the folds of the recall scores are greater than
        the center - (spread * tolerance)
        False if the folds for the recall scores are less than
        the center - (spread * tolerance)
        """
        average = self.reset_average(average)
        scores = self.recall_cv(cv, average=average)
        return self._anomaly_detection(scores, tolerance, method)

    def spread_cross_val_f1_anomaly_detection(self, tolerance,
                                              method="mean",
                                              cv=10,
                                              average='binary'):
        """
        This is a somewhat intelligent stragey,
        it generates the k fold (cross validation) f1 scores, 
        if any of the k folds score less than the center - (spread * tolerance),
        then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance modifier for how far below the 
          center the score can be before a false is returned
        method : string
          see describe for more details.
          * mean : the center is the mean, the spread is standard
                   deviation.
          * median : the center is the median, the spread is
                     the interquartile range.
          * trimean : the center is the trimean, the spread is
                      trimean absolute deviation.
        average : string
          how to calculate the f1 score
        
        Returns
        -------
        True if all the folds of the f1 scores are greater than
        the center - (spread * tolerance)
        False if the folds for the f1 scores are less than
        the center - (spread * tolerance)
        """
        average = self.reset_average(average)
        scores = self.f1_cv(cv, average=average)
        return self._anomaly_detection(scores, tolerance, method)

    def spread_cross_val_roc_auc_anomaly_detection(self, tolerance,
                                                   method="mean",
                                                   cv=10,
                                                   average='micro'):
        """
        This is a somewhat intelligent stragey,
        it generates the k fold (cross validation) roc auc scores, 
        if any of the k folds score less than the center - (spread * tolerance),
        then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance modifier for how far below the 
          center the score can be before a false is returned
        method : string
          see describe for more details.
          * mean : the center is the mean, the spread is standard
                   deviation.
          * median : the center is the median, the spread is
                     the interquartile range.
          * trimean : the center is the trimean, the spread is
                      trimean absolute deviation.
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the folds of the roc auc scores are greater than
        the center - (spread * tolerance)
        False if the folds for the roc auc scores are less than
        the center - (spread * tolerance)
        """
        self.roc_auc_exception()
        scores = self.roc_auc_cv(cv, average=average)
        return self._anomaly_detection(scores, tolerance, method)

    def spread_cross_val_classifier_testing(self,
                                            precision_tolerance: float,
                                            recall_tolerance: float,
                                            f1_tolerance: float,
                                            method="mean",
                                            cv=10, average='binary'):
        """
        This is a somewhat intelligent stragey,
        it generates the k fold (cross validation) the following scores:
        * precision scores,  
        * recall scores
        * f1 scores
        if any of the k folds score less than the center - (spread * tolerance),
        then False is returned.
        
        Parameters
        ----------
        tolerance : float
          the tolerance modifier for how far below the 
          center the score can be before a false is returned
        method : string
          see describe for more details.
          * mean : the center is the mean, the spread is standard
                   deviation.
          * median : the center is the median, the spread is
                     the interquartile range.
          * trimean : the center is the trimean, the spread is
                      trimean absolute deviation.
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the folds of the precision, recall, f1 scores 
        are greater than the center - (spread * tolerance)
        False if the folds for the precision, recall, f1 scores
        are less than the center - (spread * tolerance)
        """
        average = self.reset_average(average)
        precision_test = self.spread_cross_val_precision_anomaly_detection(
            precision_tolerance, method=method, cv=cv, average=average)
        recall_test = self.spread_cross_val_recall_anomaly_detection(
            recall_tolerance, method=method, cv=cv, average=average)
        f1_test = self.spread_cross_val_f1_anomaly_detection(
            f1_tolerance, method=method, cv=cv, average=average)
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def precision_lower_boundary_per_class(self,
                                           lower_boundary: dict,
                                           average='binary'):
        """
        This is a slightly less naive stragey,
        it checks the precision score, 
        Each class is boundary is mapped to the class via a dictionary
        allowing for different lower boundaries, per class.
        if any of the classes are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : dict
          the lower boundary for each class' 
          precision score
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the classes of the precision scores are 
        greater than the lower_boundary
        False if the classes for the precision scores are 
        less than the lower_boundary
        """
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, self.precision_score, lower_boundary)

    def recall_lower_boundary_per_class(self,
                                        lower_boundary: dict,
                                        average='binary'):
        """
        This is a slightly less naive stragey,
        it checks the recall score, 
        Each class is boundary is mapped to the class via a dictionary
        allowing for different lower boundaries, per class.
        if any of the classes are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : dict
          the lower boundary for each class' 
          recall score
        average : string
          how to calculate the recall
        
        Returns
        -------
        True if all the classes of the recall scores are 
        greater than the lower_boundary
        False if the classes for the recall scores are 
        less than the lower_boundary
        """
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, recall_score, lower_boundary)
    
    def f1_lower_boundary_per_class(self,
                                    lower_boundary: dict,
                                    average='binary'):
        """
        This is a slightly less naive stragey,
        it checks the f1 score, 
        Each class is boundary is mapped to the class via a dictionary
        allowing for different lower boundaries, per class.
        if any of the classes are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : dict
          the lower boundary for each class' f1 score
        average : string
          how to calculate the f1
        
        Returns
        -------
        True if all the classes of the f1 scores are 
        greater than the lower_boundary
        False if the classes for the f1 scores are 
        less than the lower_boundary
        """
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, f1_score, lower_boundary)

    def roc_auc_lower_boundary_per_class(self,
                                         lower_boundary: dict,
                                         average='micro'):
        """
        This is a slightly less naive stragey,
        it checks the roc auc score, 
        Each class is boundary is mapped to the class via a dictionary
        allowing for different lower boundaries, per class.
        if any of the classes are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        lower_boundary : dict
          the lower boundary for each class' roc auc score
        average : string
          how to calculate the roc auc
        
        Returns
        -------
        True if all the classes of the roc auc scores are 
        greater than the lower_boundary
        False if the classes for the roc auc scores are 
        less than the lower_boundary
        """
        self.roc_auc_exception()
        roc_auc_score = partial(self.roc_auc_score, average=average)
        y_pred = self.clf.predict(self.X)
        return self._per_class(y_pred, roc_auc_score, lower_boundary)

    def classifier_testing_per_class(self,
                                     precision_lower_boundary: dict,
                                     recall_lower_boundary: dict,
                                     f1_lower_boundary: dict,
                                     average='binary'):
        """
        This is a slightly less naive stragey,
        it checks the:
        * precision score per class, 
        * recall score per class,
        * f1 score per class
        Each class is boundary is mapped to the class via a dictionary
        allowing for different lower boundaries, per class.
        if any of the classes are less than the lower boundary,
        then False is returned.
        
        Parameters
        ----------
        precision_lower_boundary : dict
          the lower boundary for each class' precision score
        
        recall_lower_boundary : dict
          the lower boundary for each class' recall score
        
        f1_lower_boundary : dict
          the lower boundary for each class' f1 score
        
        average : string
          how to calculate the precision
        
        Returns
        -------
        True if all the classes of the precision scores are 
        greater than the lower_boundary
        False if the classes for the precision scores are 
        less than the lower_boundary
        """
        precision_test = self.precision_lower_boundary_per_class(
            precision_lower_boundary
        )
        recall_test = self.recall_lower_boundary_per_class(
            recall_lower_boundary
        )
        f1_test = self.f1_lower_boundary_per_class(
            f1_lower_boundary
        )
        if precision_test and recall_test and f1_test:
            return True
        else:
            return False

    def run_time_stress_test(self,
                             sample_sizes: list,
                             max_run_times: list):
        """
        This is a performance test to ensure that the model
        runs fast enough.
        
        Paramters
        ---------
        sample_sizes : list
          the size of each sample to test for 
          doing a prediction, each sample size is an integer
        
        max_run_times : list
          the maximum time in seconds that
          each sample should take to predict, at a maximum.
        
        Returns
        -------
        True if all samples predict within the maximum allowed
        time.
        False otherwise.
        """
        for index, sample_size in enumerate(sample_sizes):
            data = self.X.sample(sample_size, replace=True)
            start_time = time.time()
            self.clf.predict(data)
            model_run_time = time.time() - start_time
            if model_run_time > max_run_times[index]:
                return False
        return True

class ClassifierComparison(FixedClassificationMetrics):
    def __init__(self,
                 clf_one,
                 clf_two,
                 test_data,
                 target_name,
                 column_names):
        self.clf_one = clf_one
        self.clf_two = clf_two
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]
        self.classes = set(self.y)

    def is_binary(self):
        num_classes = len(set(self.classes))
        if num_classes == 2:
            return True
        return False
    
    def roc_auc_exception(self):
        if not self.is_binary():
            raise Exception("roc_auc is only defined for binary classifiers")

    def reset_average(self, average):
        if not self.is_binary() and average == 'binary':
            return 'micro'
        return average

    def two_model_prediction_run_time_stress_test(self, sample_sizes):
        for sample_size in sample_sizes:
            data = self.X.sample(sample_size, replace=True)
            start_time = time.time()
            self.clf_one.predict(data)
            model_one_run_time = time.time() - start_time
            start_time = time.time()
            self.clf_two.predict(data)
            model_two_run_time = time.time() - start_time
            # we assume model one should be faster than model two
            if model_one_run_time > model_two_run_time:
                return False
        return True
    
    def precision_per_class(self, clf, average="binary"):
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        y_pred = clf.predict(self.X)
        precision = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            precision[klass] = precision_score(y_class, y_pred_class) 
        return precision

    def recall_per_class(self, clf, average="binary"):
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        y_pred = clf.predict(self.X)
        recall = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            recall[klass] = recall_score(y_class, y_pred_class)
        return recall

    def f1_per_class(self, clf, average="binary"):
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        y_pred = clf.predict(self.X)
        f1 = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            f1[klass] = f1_score(y_class, y_pred_class)
        return f1

    def roc_auc_per_class(self, clf, average="micro"):
        self.roc_auc_exception()
        roc_auc_score = partial(self.roc_auc_score, average=average)
        y_pred = clf.predict(self.X)
        roc_auc = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            roc_auc[klass] = roc_auc_score(y_class, y_pred_class)
        return roc_auc

    def _precision_recall_f1_result(self,
                                    precision_one_test,
                                    precision_two_test,
                                    recall_one_test,
                                    recall_two_test,
                                    f1_one_test,
                                    f1_two_test):
        for klass in precision_one_test:
            precision_result =  precision_one_test[klass] < precision_two_test[klass]
            recall_result = recall_one_test[klass] < recall_two_test[klass]
            f1_result = f1_one_test[klass] < f1_two_test[klass]
            if precision_result or recall_result or f1_result:
                return False
        return True

    def _precision_recall_f1_roc_auc_result(self,
                                            precision_one_test,
                                            precision_two_test,
                                            recall_one_test,
                                            recall_two_test,
                                            f1_one_test,
                                            f1_two_test,
                                            roc_auc_one_test,
                                            roc_auc_two_test):
        for klass in precision_one_test:
            precision_result =  precision_one_test[klass] < precision_two_test[klass]
            recall_result = recall_one_test[klass] < recall_two_test[klass]
            f1_result = f1_one_test[klass] < f1_two_test[klass]
            roc_auc_result = roc_auc_one_test[klass] < roc_auc_two_test[klass]
            if precision_result or recall_result or f1_result or roc_auc_result:
                return False
        return True

    def two_model_classifier_testing(self, average="binary"):
        average = self.reset_average(average)
        precision_one_test = self.precision_per_class(self.clf_one, average=average)
        recall_one_test = self.recall_per_class(self.clf_one, average=average)
        f1_one_test = self.f1_per_class(self.clf_one, average=average)
        precision_two_test = self.precision_per_class(self.clf_two, average=average)
        recall_two_test = self.recall_per_class(self.clf_two, average=average)
        f1_two_test = self.f1_per_class(self.clf_two, average=average)
        if self.is_binary():
            if average == 'binary':
                average = 'micro'
            roc_auc_one_test = self.roc_auc_per_class(self.clf_one, average=average)
            roc_auc_two_test = self.roc_auc_per_class(self.clf_two, average=average)
            return self._precision_recall_f1_roc_auc_result(precision_one_test,
                                                            precision_two_test,
                                                            recall_one_test,
                                                            recall_two_test,
                                                            f1_one_test,
                                                            f1_two_test,
                                                            roc_auc_one_test,
                                                            roc_auc_two_test)
        else:
            self._precision_recall_f1_result(precision_one_test,
                                             precision_two_test,
                                             recall_one_test,
                                             recall_two_test,
                                             f1_one_test,
                                             f1_two_test)
        
    def cross_val_precision_per_class(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        precision = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            precision[klass] = precision_score(y_class, y_pred_class) 
        return precision

    def cross_val_recall_per_class(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        recall = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            recall[klass] = recall_score(y_class, y_pred_class)
        return recall

    def cross_val_f1_per_class(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        f1 = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            f1[klass] = f1_score(y_class, y_pred_class)
        return f1

    def cross_val_roc_auc_per_class(self, clf, cv=3, average="micro"):
        self.roc_auc_exception()
        roc_auc_score = partial(self.roc_auc_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        roc_auc = {}
        for klass in self.classes:
            y_pred_class = np.take(y_pred, self.y[self.y == klass].index, axis=0)
            y_class = self.y[self.y == klass]
            roc_auc[klass] = roc_auc_score(y_class, y_pred_class)
        return roc_auc

    def cross_val_per_class_two_model_classifier_testing(self, cv=3, average="binary"):
        average = self.reset_average(average)
        precision_one_test = self.cross_val_precision_per_class(self.clf_one,
                                                                cv=cv, average=average)
        recall_one_test = self.cross_val_recall_per_class(self.clf_one,
                                                          cv=cv, average=average)
        f1_one_test = self.cross_val_f1_per_class(self.clf_one,
                                                  cv=cv, average=average)
        precision_two_test = self.cross_val_precision_per_class(self.clf_two,
                                                                cv=cv, average=average)
        recall_two_test = self.cross_val_recall_per_class(self.clf_two,
                                                          cv=cv, average=average)
        f1_two_test = self.cross_val_f1_per_class(self.clf_two,
                                                  cv=cv, average=average)
        if self.is_binary():
            if average == 'binary':
                average = 'micro'
            roc_auc_one_test = self.roc_auc_per_class(self.clf_one, average=average)
            roc_auc_two_test = self.roc_auc_per_class(self.clf_two, average=average)
            return self._precision_recall_f1_roc_auc_result(precision_one_test,
                                                            precision_two_test,
                                                            recall_one_test,
                                                            recall_two_test,
                                                            f1_one_test,
                                                            f1_two_test,
                                                            roc_auc_one_test,
                                                            roc_auc_two_test)
        else:
            self._precision_recall_f1_result(precision_one_test,
                                             precision_two_test,
                                             recall_one_test,
                                             recall_two_test,
                                             f1_one_test,
                                             f1_two_test)

    def cross_val_precision(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        precision_score = partial(self.precision_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return precision_score(self.y, y_pred) 

    def cross_val_recall(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        recall_score = partial(self.recall_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return recall_score(self.y, y_pred)

    def cross_val_f1(self, clf, cv=3, average="binary"):
        average = self.reset_average(average)
        f1_score = partial(self.f1_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return f1_score(self.y, y_pred)

    def cross_val_roc_auc(self, clf, cv=3, average="micro"):
        self.roc_auc_exception()
        roc_auc_score = partial(self.roc_auc_score, average=average)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=cv)
        return roc_auc_score(self.y, y_pred)

    def cross_val_two_model_classifier_testing(self, cv=3, average="binary"):
        average = self.reset_average(average)
        precision_one_test = self.cross_val_precision(self.clf_one,
                                                      cv=cv, average=average)
        recall_one_test = self.cross_val_recall(self.clf_one,
                                                cv=cv, average=average)
        f1_one_test = self.cross_val_f1(self.clf_one,
                                        cv=cv, average=average)
        precision_two_test = self.cross_val_precision(self.clf_two,
                                                      cv=cv, average=average)
        recall_two_test = self.cross_val_recall(self.clf_two,
                                                cv=cv, average=average)
        f1_two_test = self.cross_val_f1(self.clf_two,
                                        cv=cv, average=average)
        precision_result =  precision_one_test < precision_two_test
        recall_result = recall_one_test < recall_two_test
        f1_result = f1_one_test < f1_two_test
        if self.is_binary():
            if average == 'binary':
                average = 'micro'
            roc_auc_one_test = self.cross_val_roc_auc(self.clf_one,
                                                      cv=cv, average=average)
            roc_auc_two_test = self.cross_val_roc_auc(self.clf_two,
                                                      cv=cv, average=average)
            roc_auc_result = roc_auc_one_test < roc_auc_two_test
            if precision_result or recall_result or f1_result or roc_auc_result:
                return False
            else:
                return True
        else:
            if precision_result or recall_result or f1_result:
                return False
            else:
                return True
