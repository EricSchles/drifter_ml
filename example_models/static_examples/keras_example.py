from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import numpy as np
import joblib

# Function to create model, required for KerasClassifier
def create_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=3, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

# fix random seed for reproducibility
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

# split into input (X) and output (Y) variables
# create model
clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
X = df[["A", "B", "C"]]
clf.fit(X, df["target"])
joblib.dump(clf, "model.joblib")
df.to_csv("data.csv")
