import joblib
import json
from sklearn import metrics
import numpy as np
import time
from sklearn import neighbors
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn import cluster

class KmeansClustering():
    def __init__(self,
                 new_data,
                 historical_data,
                 column_names,
                 target_name):
        self.column_names = column_names
        self.target_name = target_name
        self.new_data = new_data
        self.historical_data = historical_data

    def kmeans_clusters(self, n_clusters, data):
        k_means = cluster.KMeans(n_clusters=n_clusters)
        k_means.fit(data)
        return k_means.predict(data)

    def kmeans_scorer(self, metric, min_similarity):
        for k in range(2, 12):
            new_data = self.new_data[self.column_names]
            historical_data = self.historical_data[self.column_names]
            new_data_clusters = self.kmeans_clusters(k, new_data)
            historical_clusters = self.kmeans_clusters(k, historical_data)
            score = metric(
                new_data_clusters, historical_data_clusters)
            if score < min_similarity:
                return False
        return True
    
    def mutual_info_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.adjusted_mutual_info_score,
            min_similarity
        )

    def adjusted_rand_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.adjusted_rand_score,
            min_similarity
        )

    def completeness_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.completeness_score,
            min_similarity
        )

    def fowlkes_mallows_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.fowlkes_mallows_score,
            min_similarity
        )

    def homogeneity_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.homogeneity_score,
            min_similarity
        )

    def v_measure_kmeans_scorer(self, min_similarity):
        return self.kmeans_scorer(
            metrics.v_measure_score,
            min_similarity
        )

    def unsupervised_kmeans_score_clustering(self, min_similarity):
        return all([
            self.v_measure_kmeans_scorer(min_similarity),
            self.homogeneity_kmeans_scorer(min_similarity),
            self.fowlkes_mallows_kmeans_scorer(min_similarity),
            self.completeness_kmeans_scorer(min_similarity),
            self.adjusted_rand_kmeans_scorer(min_similarity),
            self.mutual_info_kmeans_scorer(min_similarity),
        ])

class DBscanClustering():
    def __init__(self,
                 new_data,
                 historical_data,
                 column_names,
                 target_name):
        self.column_names = column_names
        self.target_name = target_name
        self.new_data = new_data
        self.historical_data = historical_data

    def dbscan_clusters(self, data):
        dbscan = cluster.DBSCAN()
        return dbscan.fit_predict(data)
    
    def dbscan_scorer(self, metric, min_similarity):
        for k in range(2, 12):
            new_data = self.new_data[self.column_names]
            historical_data = self.historical_data[self.column_names]
            new_data_clusters = self.dbscan_clusters(new_data)
            historical_clusters = self.dbscan_clusters(historical_data)
            score = metric(
                new_data_clusters, historical_data_clusters)
            if score < min_similarity:
                return False
        return True

    def mutual_info_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.adjusted_mutual_info_score,
            min_similarity
        )

    def adjusted_rand_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.adjusted_rand_score,
            min_similarity
        )

    def completeness_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.completeness_score,
            min_similarity
        )

    def fowlkes_mallows_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.fowlkes_mallows_score,
            min_similarity
        )

    def homogeneity_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.homogeneity_score,
            min_similarity
        )

    def v_measure_dbscan_scorer(self, min_similarity):
        return self.dbscan_scorer(
            metrics.v_measure_score,
            min_similarity
        )

    def unsupervised_dbscan_score_clustering(self, min_similarity):
        return all([
            self.v_measure_dbscan_scorer(min_similarity),
            self.homogeneity_dbscan_scorer(min_similarity),
            self.fowlkes_mallows_dbscan_scorer(min_similarity),
            self.completeness_dbscan_scorer(min_similarity),
            self.adjusted_rand_dbscan_scorer(min_similarity),
            self.mutual_info_dbscan_scorer(min_similarity),
        ])

class KnnClustering():
    def __init__(self,
                 new_data,
                 historical_data,
                 column_names,
                 target_name):
        self.column_names = column_names
        self.target_name = target_name
        self.new_data = new_data
        self.historical_data = historical_data

    def reg_supervised_clustering(self, data):
        k_measures = []
        X = data[self.column_names]
        y = data[self.target_name]
        for k in range(2, 12):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(X, y)
            y_pred = knn.predict(X)
            k_measures.append((k, metrics.mean_squared_error(y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def reg_supervised_similar_clustering(self, absolute_distance):
        historical_k = reg_clustering(self.historical_data)
        new_k = reg_clustering(self.new_data)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True

    def cls_supervised_clustering(self, data):
        k_measures = []
        X = data[self.column_names]
        y = data[self.target_name]
        for k in range(2, 12):
            knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            knn.fit(X, y)
            y_pred = knn.predict(X)
            k_measures.append((k, metrics.mean_squared_error(y, y_pred)))
        sorted_k_measures = sorted(k_measures, key=lambda t:t[1])
        lowest_mse = sorted_k_measures[0]
        best_k = lowest_mse[0]
        return best_k

    def cls_supervised_similiar_clustering(self, absolute_distance):
        historical_k = cls_clustering(self.historical_data)
        new_k = cls_clustering(self.new_data)
        if abs(historical_k - new_k) > absolute_distance:
            return False
        else:
            return True

class StructuralData(KnnClustering,
                     DBscanClustering,
                     KmeansClustering):
    def __init__(self,
                 new_data,
                 historical_data,
                 column_names,
                 target_name):
        self.column_names = column_names
        self.target_name = target_name
        self.new_data = new_data
        self.historical_data = historical_data

