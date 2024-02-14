

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


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


def make_ratio(X):
    return X[:, [0]] / X[:, [1]]

def feature_names_out(transformer=None, feature_names_in=None):
    return ["ratio"]

def make_ratio_pipline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(make_ratio,
                            feature_names_out=feature_names_out),
        StandardScaler()
    )