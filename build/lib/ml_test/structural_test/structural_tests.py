import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_val_score

class StructuralData():
    def __init__(self, metadata, data_filename):
        metadata, column_names, target_name, test_data = self.get_parameters(
            metadata, data_filename)
        self.data_filename
        self.metadata = metadata
        self.column_names = column_names
        self.target_name = target_name
        self.test_data = test_data
        self.y = test_data[target_name]
        self.X = test_data[column_names]

    def get_parameters(self, metadata, data_filename):
        metadata = json.load(open(clf_metadata, "r"))
        column_names = metadata["column_names"]
        target_name = metadata["target_name"]
        test_data = pd.read_csv(data_name)
        return metadata, column_names, target_name, test_data

    def reg_clustering(self, data, columns, target):
        k_measures = []
        for k in range(2, 12):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.X, self.y)
            y_pred = knn.predict(self.X)
            k_measures.append((k, metrics.mean_squared_error(self.y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def reg_similar_clustering(self, absolute_distance, new_data, historical_data, column_names, target_name):
        historical_k = reg_clustering(historical_data, column_names, target_name)
        new_k = reg_clustering(new_data, column_names, target_name)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True

    # this was never updated
    def cls_clustering(self):
        k_measures = []
        for k in range(2, 12):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.X, self.y)
            y_pred = knn.predict(self.X)
            k_measures.append((k, metrics.mean_squared_error(self.y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def cls_similiar_clustering(absolute_distance, new_data, historical_data, column_names, target_name):
        historical_k = cls_clustering(historical_data, column_names, target_name)
        new_k = cls_clustering(new_data, column_names, target_name)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True
