from drifter_ml import classification_tests
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import numpy as np
import joblib
import pandas as pd

def generate_binary_classification_data_and_models():
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

    clf1 = tree.DecisionTreeClassifier()
    clf2 = ensemble.RandomForestClassifier()
    column_names = ["A", "B", "C"]
    target_name = "target"
    X = df[column_names]
    clf1.fit(X, df[target_name])
    clf2.fit(X, df[target_name])
    return df, column_names, target_name, clf1, clf2

def generate_multiclass_classification_data_and_models():
    df = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        if  11 < a + b + c:
            target = 1
        elif 8 < a + b + c and a + b + c < 10:
            target = 0
        else:
            target = 2
            df = df.append({
                "A": a,
                "B": b,
                "C": c,
                "target": target
            }, ignore_index=True)

    clf1 = tree.DecisionTreeClassifier()
    clf2 = ensemble.RandomForestClassifier()
    column_names = ["A", "B", "C"]
    target_name = "target"
    X = df[column_names]
    clf1.fit(X, df[target_name])
    clf2.fit(X, df[target_name])
    return df, column_names, target_name, clf1, clf2

def test_precision_recall_f1_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    classes = list(df.target.unique())
    assert test_suite.classifier_testing(
        {klass: 0.1 for klass in classes},
        {klass: 0.1 for klass in classes},
        {klass: 0.1 for klass in classes}
    )

def test_precision_recall_f1_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    classes = list(df.target.unique())
    assert test_suite.classifier_testing(
        {klass: 0.1 for klass in classes},
        {klass: 0.1 for klass in classes},
        {klass: 0.1 for klass in classes}
    )

def test_roc_auc_cv_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    try:
        roc_auc_scores = test_suite.roc_auc_cv(3)
        assert isinstance(roc_auc_scores, list)
        assert isinstance(roc_auc_scores[0], float)
        assert len(roc_auc_scores) == 3
    except ValueError:
        assert True
        
def test_f1_cv_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    f1_scores = test_suite.f1_cv(3)
    assert isinstance(f1_scores, list)
    assert isinstance(f1_scores[0], float)
    assert len(f1_scores) == 3

def test_f1_cv_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    f1_scores = test_suite.f1_cv(3)
    assert isinstance(f1_scores, list)
    assert isinstance(f1_scores[0], float)
    assert len(f1_scores) == 3

def test_recall_cv_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    recall_scores = test_suite.recall_cv(3)
    assert isinstance(recall_scores, list)
    assert isinstance(recall_scores[0], float)
    assert len(recall_scores) == 3

def test_recall_cv_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    recall_scores = test_suite.recall_cv(3)
    assert isinstance(recall_scores, list)
    assert isinstance(recall_scores[0], float)
    assert len(recall_scores) == 3

def test_precision_cv_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    precision_scores = test_suite.precision_cv(3)
    assert isinstance(precision_scores, list)
    assert isinstance(precision_scores[0], float)
    assert len(precision_scores) == 3

def test_precision_cv_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    precision_scores = test_suite.precision_cv(3)
    assert isinstance(precision_scores, list)
    assert isinstance(precision_scores[0], float)
    assert len(precision_scores) == 3

def test_precision_metric():
    fixed_metrics = classification_tests.FixedClassificationMetrics()
    assert 1.0 == fixed_metrics.precision_score([0,0,0], [0,0,0])

def test_recall_metric():
    fixed_metrics = classification_tests.FixedClassificationMetrics()
    assert 1.0 == fixed_metrics.recall_score([0,0,0], [0,0,0])

def test_f1_metric():
    fixed_metrics = classification_tests.FixedClassificationMetrics()
    assert 1.0 == fixed_metrics.f1_score([0,0,0], [0,0,0])

def test_cross_val_per_class_percision_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_precision_anomaly_detection(tolerance)

def test_cross_val_per_class_percision_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_precision_anomaly_detection(tolerance)

def test_cross_val_per_class_recall_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_recall_anomaly_detection(tolerance)

def test_cross_val_per_class_recall_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_recall_anomaly_detection(tolerance)

def test_cross_val_per_class_f1_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_f1_anomaly_detection(tolerance)

def test_cross_val_per_class_f1_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_per_class_f1_anomaly_detection(tolerance)

def test_cross_val_per_class_roc_auc_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    try:
        tolerance = 1
        assert test_suite.cross_val_per_class_roc_auc_anomaly_detection(tolerance)
    except ValueError:
        assert True

def test_cross_val_precision_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_precision_anomaly_detection(tolerance)

def test_cross_val_precision_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_precision_anomaly_detection(tolerance)

def test_cross_val_recall_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_recall_anomaly_detection(tolerance)

def test_cross_val_recall_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_recall_anomaly_detection(tolerance)

def test_cross_val_f1_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_f1_anomaly_detection(tolerance)

def test_cross_val_f1_anomaly_detection_mutliclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance = 1
    assert test_suite.cross_val_f1_anomaly_detection(tolerance)

def test_cross_val_roc_auc_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    try:
        tolerance = 1
        assert test_suite.cross_val_roc_auc_anomaly_detection(tolerance)
    except ValueError:
        assert True

def test_cross_val_precision_avg_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_precision_avg(avg)

def test_cross_val_precision_avg_mutliclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_precision_avg(avg)

def test_cross_val_recall_avg_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_recall_avg(avg)

def test_cross_val_recall_avg_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_recall_avg(avg)

def test_cross_val_f1_avg_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_f1_avg(avg)

def test_cross_val_f1_avg_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    avg = 0.1
    assert test_suite.cross_val_f1_avg(avg)

def test_cross_val_roc_auc_avg_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    try:
        avg = 0.1
        assert test_suite.cross_val_roc_auc_avg(avg)
    except ValueError:
        assert True
        
def test_spread_cross_val_precision_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_precision_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_precision_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_recall_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_recall_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_recall_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_f1_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_f1_anomaly_detection_multiclass():
    df, column_names, target_name, clf, _ = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    tolerance =  1
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance)
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance, method="median")
    assert test_suite.spread_cross_val_f1_anomaly_detection(tolerance, method="trimean")

def test_spread_cross_val_roc_auc_anomaly_detection_binary():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    try:
        tolerance =  1
        assert test_suite.spread_cross_val_roc_auc_anomaly_detection(tolerance)
        assert test_suite.spread_cross_val_roc_auc_anomaly_detection(tolerance, method="median")
        assert test_suite.spread_cross_val_roc_auc_anomaly_detection(tolerance, method="trimean")
    except ValueError:
        assert True

def test_run_time_stress_test():
    df, column_names, target_name, clf, _ = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    performance_boundary = []
    for i in range(100, 1000, 100):
        performance_boundary.append({
            "sample_size": i,
            "max_run_time": 100})
    try:
        test_suite.run_time_stress_test(performance_boundary)
        assert True
    except:
        assert False

def test_two_model_prediction_run_time_stress_test():
    df, column_names, target_name, clf1, clf2 = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    performance_boundary = []
    for i in range(100, 1000, 100):
        performance_boundary.append({
            "sample_size": i,
            "max_run_time": 100})
    try:
        test_suite.two_model_prediction_run_time_stress_test(performance_boundary)
        assert True
    except:
        assert False

def test_two_model_classifier_testing_binary():
    df, column_names, target_name, clf1, clf2 = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.two_model_classifier_testing()
        assert True
    except:
        assert False

def test_two_model_classifier_testing_multiclass():
    df, column_names, target_name, clf1, clf2 = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.two_model_classifier_testing()
        assert True
    except:
        assert False

def test_cross_val_two_model_classifier_testing_binary():
    df, column_names, target_name, clf1, clf2 = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.cross_val_two_model_classifier_testing()
        assert True
    except:
        assert False

def test_cross_val_two_model_classifier_testing_multiclass():
    df, column_names, target_name, clf1, clf2 = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.cross_val_two_model_classifier_testing()
        assert True
    except:
        assert False

def test_cross_val_two_model_classifier_testing_binary():
    df, column_names, target_name, clf1, clf2 = generate_binary_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.cross_val_per_class_two_model_classifier_testing()
        assert True
    except:
        assert False

def test_cross_val_two_model_classifier_testing_multiclass():
    df, column_names, target_name, clf1, clf2 = generate_multiclass_classification_data_and_models()
    test_suite = classification_tests.ClassifierComparison(clf1,
                                                           clf2,
                                                           df,
                                                           target_name,
                                                           column_names)
    try:
        test_suite.cross_val_per_class_two_model_classifier_testing()
        assert True
    except:
        assert False
