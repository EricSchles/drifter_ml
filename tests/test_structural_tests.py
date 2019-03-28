from drifter_ml import structural_tests
import numpy as np
import pandas as pd

def generate_classification_data_and_models():
    new_data = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        if a + b + c > 11:
            target = 1
        else:
            target = 0
            new_data = new_data.append({
                "A": a,
                "B": b,
                "C": c,
                "target": target
            }, ignore_index=True)

    historical_data = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        if a + b + c > 11:
            target = 1
        else:
            target = 0
            historical_data = historical_data.append({
                "A": a,
                "B": b,
                "C": c,
                "target": target
            }, ignore_index=True)

    column_names = ["A", "B", "C"]
    target_name = "target"
    return new_data, historical_data, column_names, target_name

def generate_regression_data_and_models():
    new_data = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        target = a + b + c
        new_data = new_data.append({
            "A": a,
            "B": b,
            "C": c,
            "target": target
        }, ignore_index=True)

    historical_data = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        target = a + b + c
        historical_data = historical_data.append({
            "A": a,
            "B": b,
            "C": c,
            "target": target
        }, ignore_index=True)

    column_names = ["A", "B", "C"]
    target_name = "target"
    return new_data, historical_data, column_names, target_name

def generate_unsupervised_data():
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

def test_mutual_info_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.mutual_info_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_adjusted_rand_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.adjusted_rand_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_completeness_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.completeness_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_fowlkes_mallows_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.fowlkes_mallows_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_homogeneity_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.homogeneity_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_v_measure_kmeans_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.v_measure_kmeans_scorer(min_similarity)
        assert True
    except:
        assert False

def test_mutual_info_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.mutual_info_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_adjusted_rand_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.adjusted_rand_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_completeness_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.completeness_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_fowlkes_mallows_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.fowlkes_mallows_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_homogeneity_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.homogeneity_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_v_measure_dbscan_scorer():
    new_data, historical_data = generate_unsupervised_data()
    columns = ["similar_normal", "different_normal",
               "similar_gamma", "different_gamma"]
    target = ''
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 columns,
                                                 target)
    try:
        min_similarity = 0.5
        test_suite.v_measure_dbscan_scorer(min_similarity)
        assert True
    except:
        assert False

def test_reg_supervised_similar_clustering():
    new_data, historical_data, column_names, target_name = generate_regression_data_and_models()

    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 column_names,
                                                 target_name)
    try:
        absolute_distance = 2
        test_suite.reg_supervised_similar_clustering(absolute_distance)
        assert True
    except:
        assert False

def test_reg_supervised_similar_clustering():
    new_data, historical_data, column_names, target_name = generate_classification_data_and_models()
    test_suite = structural_tests.StructuralData(new_data,
                                                 historical_data,
                                                 column_names,
                                                 target_name)
    try:
        absolute_distance = 2
        test_suite.cls_supervised_similar_clustering(absolute_distance)
        assert True
    except:
        assert False

