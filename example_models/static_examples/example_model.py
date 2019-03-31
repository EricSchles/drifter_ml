from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import pandas as pd
import numpy as np
import joblib
import code
import json

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
X = df[["A", "B", "C"]]
clf1.fit(X, df["target"])
clf2.fit(X, df["target"])
#code.interact(local=locals())
joblib.dump(clf1, "model1.joblib")
joblib.dump(clf2, "model1.joblib")
json.dump({
    "column_names": ["A", "B", "C"],
    "target_name": "target"
    }, open("model_metadata.json", "w"))
df.to_csv("data.csv")
