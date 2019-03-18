from drifter_ml import classification_tests
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import numpy as np
import joblib
import pandas as pd


def generate_classification_data_and_models():
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


def test_precision_recall_f1_basic():
    df, column_names, target_name, clf, _ = generate_classification_data_and_models()
    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    classes = list(df.target.unique())
    assert test_suite.classifier_testing(
        {klass: 0.000000001 for klass in classes},
        {klass: 0.00000001 for klass in classes},
        {klass: 0.00000001 for klass in classes}
    )
    
    

