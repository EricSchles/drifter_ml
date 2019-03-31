from drifter_ml import columnar_tests
import numpy as np
import pandas as pd

def generate_data():
    new_data = pd.DataFrame()
    historical_data = pd.DataFrame()
    new_data["similar_normal"] = np.random.normal(0, 10, size=1000)
    historical_data["similar_normal"] = np.random.normal(0, 10, size=1000)
    new_data["different_normal"] = np.random.normal(1000, 250, size=1000)
    historical_data["different_normal"] = np.random.normal(5, 17, size=1000)
    new_data["random"] = np.random.random(size=1000)
    historical_data["random"] = np.random.random(size=1000)
    new_data["similar_gamma"] = np.random.gamma(1, 2, size=1000)
    historical_data["similar_gamma"] = np.random.gamma(1, 2, size=1000)
    new_data["different_gamma"] = np.random.gamma(7.5, 0, size=1000)
    historical_data["different_gamma"] = np.random.gamma(2, 4, size=1000)
    return new_data, historical_data

def test_mean_similarity():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.mean_similarity("similar_normal")
        assert True
    except:
        assert False
    
def test_median_similarity():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.median_similarity("similar_normal")
        assert True
    except:
        assert False


def test_trimean_similarity():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.trimean_similarity("similar_normal")
        assert True
    except:
        assert False
    
def test_is_normal():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.is_normal("similar_normal")
        assert True
    except:
        assert False

def test_pearson_similar_correlation():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    correlation_lower_bound = 0.3
    try:
        test_suite.pearson_similar_correlation("similar_normal", correlation_lower_bound)
        assert True
    except:
        assert False

def test_spearman_similar_correlation():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    correlation_lower_bound = 0.3
    try:
        test_suite.spearman_similar_correlation("similar_normal", correlation_lower_bound)
        assert True
    except:
        assert False

def test_wilcoxon_similar_distribution():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.wilcoxon_similar_distribution("similar_normal")
        assert True
    except:
        assert False
        
def test_ks_2samp_similar_distribution():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.ks_2samp_similar_distribution("similar_normal")
        assert True
    except:
        assert False

def test_kruskal_similar_distribution():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.kruskal_similar_distribution("similar_normal")
        assert True
    except:
        assert False
        
def test_mann_whitney_u_similar_distribution():
    new_data, historical_data = generate_data()
    test_suite = columnar_tests.ColumnarData(new_data, historical_data)
    try:
        test_suite.mann_whitney_u_similar_distribution("similar_normal")
        assert True
    except:
        assert False
