from drifter_ml import classification_tests
import joblib
import pandas as pd
import code

def test():
    df = pd.read_csv("data.csv")
    column_names = ["A", "B", "C"]
    target_name = "target"
    clf = joblib.load("model1.joblib")

    test_suite = classification_tests.ClassificationTests(clf,
                                                          df,
                                                          target_name,
                                                          column_names)
    classes = list(df.target.unique())
    assert test_suite.classifier_testing(
        {klass: 0.9 for klass in classes},
        {klass: 0.9 for klass in classes},
        {klass: 0.9 for klass in classes}
    )
    
    

