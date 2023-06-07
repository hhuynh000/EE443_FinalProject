# This is the baseline code for merging the fragment tracklet using clustering based on appearance
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

class postprocess:
    def __init__(self,number_of_people,cluster_method):
        self.n = number_of_people
        if cluster_method == 'kmeans':
            self.cluster_method = KMeans(n_clusters=self.n, random_state=0)
        elif cluster_method == 'agglo':
            print(cluster_method)
            self.cluster_method = AgglomerativeClustering(n_clusters=None, affinity='cosine', 
                                                          linkage='single', distance_threshold=0.25)
        else:
            raise NotImplementedError
    
    def run(self,features):

        print('Start Clustering')
        self.cluster_method.fit(features)
        print('Finish Clustering')

        return self.cluster_method.labels_