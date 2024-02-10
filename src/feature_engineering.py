

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, gamma=0.1):
        self.n_clusters = n_clusters
        self.gamma = gamma
        
    def fit(self, X, y=None):
        self.cluster_centers_ = KMeans(self.n_clusters).fit(X).cluster_centers_
        return self
    
    def transform(self, X):
        return rbf_kernel(X, Y=self.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, feature_names_in=None):
        return [f"similarity_cluster_{i+1}" for i in range(self.n_clusters)]

