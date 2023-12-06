from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing


class Reducer:
    def __init__(self, intermediate_components=-1,
                 perplexity=30, num_components=-1, final_reducer='tsne'):
        self.intermediate_components = intermediate_components
        self.perplexity = perplexity
        self.num_components = num_components
        self.intermediate = intermediate_components != -1
        self.final = num_components != -1
        # exit()
        # print("HIHIHIHIHIHIHIHIHIHIHIHIHIHI: ", self.intermediate, self.final)
        
        if self.intermediate:
            self.pre_reducer = PCA(n_components=self.intermediate_components)
        
        if final_reducer == 'tsne':
            self.reducer = TSNE(n_components=self.num_components,
                                perplexity=self.perplexity,
                                metric='cosine'
                                )
        else:
            self.reducer = PCA(n_components=self.num_components)

    def reduce(self, embeddings):
        if not self.final:
            return embeddings, embeddings
        if self.intermediate:
            embeddings = self.pre_reducer.fit_transform(embeddings)
            
        return embeddings, self.reducer.fit_transform(embeddings)
        # if embeddings.shape[1] == self.num_components:
        #     return embeddings, embeddings
        # if self.intermediate:
        


class Scaler:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        
    def predict(self, embeddings):
        return self.scaler.fit_transform(embeddings)
