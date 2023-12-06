from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from model.reducer import Reducer, Scaler
from model.utils import distance_metric, construct_connectivity


class Clusterer:
    def __init__(self, method, distance, num_clusters, embedding_dim, intermediate_components=50, final_reducer='tsne'):
        self.method = method
        self.num_clusters = num_clusters
        self.distance = distance
        self.metric = distance_metric(distance)
        
        self.reducer = Reducer(num_components=embedding_dim, intermediate_components=intermediate_components, final_reducer=final_reducer)
        self.scaler = Scaler()
        
        if self.method == 'kmeans':
            print(f"Using K-Means with {num_clusters} clusters")
        elif self.method == 'dbscan':
            print(f"Using {self.distance} distance metric for DBSCAN")
        elif self.method == 'agglo':
            self.linkage = 'ward' if distance == 'euclidean' else 'average'
            print(f"Using {distance} distance metric and {self.linkage} linkage for Agglomerative Clustering")
        elif self.method == 'gaussian':
            print("Using Bayesian inference for Gaussian Mixture Model")
        elif self.method == 'ours':
            print("Using our method")
        else:
            raise ValueError('Invalid clustering method')

    def cluster(self, embeddings):
        pre_embeddings, reduced_embeddings = self.reducer.reduce(embeddings)
        print("HIHIHIHIHIHIHIHIHIHIHIHIHIHI: ", pre_embeddings.shape, reduced_embeddings.shape)
        scaled_embeddings = self.scaler.predict(reduced_embeddings)
        
        if self.method == 'ours':
            print(f"Using our method with {self.distance} distance metric and {self.num_clusters} clusters")
            birch_model = Birch(threshold=0.5, n_clusters=None)
            subcluster_labels = birch_model.fit_predict(scaled_embeddings)
            
            connectivity = construct_connectivity(scaled_embeddings,
                                                  subcluster_labels)
            
            agglo_model = AgglomerativeClustering(n_clusters=self.num_clusters,
                                                  metric=self.distance,
                                                  linkage="single",
                                                  connectivity=connectivity
                                                  )
            
            labels = agglo_model.fit_predict(scaled_embeddings)
        else:
            if self.method == 'kmeans':
                model = KMeans(n_clusters=self.num_clusters, n_init='auto')
            elif self.method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10, metric=self.metric)
            elif self.method == 'agglo':
                model = AgglomerativeClustering(n_clusters=self.num_clusters,
                                                metric=self.distance,
                                                linkage=self.linkage
                                                )
            elif self.method == 'gaussian':
                model = BayesianGaussianMixture(n_components=self.num_clusters)
            
            else:
                raise ValueError('Invalid clustering method')
        
            labels = model.fit_predict(scaled_embeddings)
        
        if self.method == 'dbscan':
            self.num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return labels, pre_embeddings
