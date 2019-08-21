from drifter_ml import regression_tests
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import numpy as np
import pandas as pd

def generate_regression_data_and_models():
    df = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        target = a + b + c
        df = df.append({
            "A": a,
            "B": b,
            "C": c,
            "target": target
        }, ignore_index=True)

    reg1 = tree.DecisionTreeRegressor()
    reg2 = ensemble.RandomForestRegressor()
    column_names = ["A", "B", "C"]
    target_name = "target"
    X = df[column_names]
    reg1.fit(X, df[target_name])
    reg2.fit(X, df[target_name])
    return df, column_names, target_name, reg1, reg2

def test_regression_basic():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        mse_upper_boundary = 10000
        mae_upper_boundary = 10000
        tse_upper_boundary = 10000
        tae_upper_boundary = 10000
        test_suite.upper_bound_regression_testing(
            mse_upper_boundary,
            mae_upper_boundary,
            tse_upper_boundary,
            tae_upper_boundary
        )
        assert True
    except:
        assert False

def test_cross_val_mse_anomaly_detection():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        mse_tolerance = 10000
        test_suite.cross_val_mse_anomaly_detection(
            mse_tolerance
        )
        assert True
    except:
        assert False

def test_cross_val_tse_anomaly_detection():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        tse_tolerance = 10000
        test_suite.cross_val_tse_anomaly_detection(
            tse_tolerance
        )
        assert True
    except:
        assert False

def test_cross_val_mae_anomaly_detection():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        
        mae_tolerance = 10000
        test_suite.cross_val_mae_anomaly_detection(
            mae_tolerance
        )
        assert True
    except:
        assert False

def test_cross_val_tae_anomaly_detection():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        
        tae_tolerance = 10000
        test_suite.cross_val_tae_anomaly_detection(
            tae_tolerance
        )
        assert True
    except:
        assert False

def test_cross_val_mse_avg():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        mse_avg = 100
        test_suite.cross_val_mse_avg(
            mse_avg
        )
        assert True
    except:
        assert False

def test_cross_val_tse_avg():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        tse_avg = 100
        test_suite.cross_val_tse_avg(
            tse_avg
        )
        assert True
    except:
        assert False

def test_cross_val_mae_avg():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        mae_avg = 100
        test_suite.cross_val_mae_avg(
            mae_avg
        )
        assert True
    except:
        assert False

def test_cross_val_tae_avg():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        tae_avg = 100
        test_suite.cross_val_tae_avg(
            tae_avg
        )
        assert True
    except:
        assert False

def test_run_time_stress_test():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    
    sample_sizes = [i for i in range(100, 1000, 100)]
    max_run_times = [100 for _ in range(len(sample_sizes))]
    try:
        test_suite.run_time_stress_test(
            sample_sizes, max_run_times
        )
        assert True
    except:
        assert False

def test_two_model_prediction_run_time_stress_test():
    df, column_names, target_name, reg1, reg2 = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionComparison(reg1,
                                                       reg2,
                                                       df,
                                                       target_name,
                                                       column_names)
    sample_sizes = [i for i in range(100, 1000, 100)]
    try:
        test_suite.two_model_prediction_run_time_stress_test(
            sample_sizes
        )
        assert True
    except:
        assert False

def test_cv_two_model_regression_testing():
    df, column_names, target_name, reg1, reg2 = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionComparison(reg1,
                                                       reg2,
                                                       df,
                                                       target_name,
                                                       column_names)
    try:
        test_suite.cv_two_model_regression_testing()
        assert True
    except:
        assert False

def test_two_model_regression_testing():
    df, column_names, target_name, reg1, reg2 = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionComparison(reg1,
                                                       reg2,
                                                       df,
                                                       target_name,
                                                       column_names)
    try:
        test_suite.two_model_regression_testing()
        assert True
    except:
        assert False
