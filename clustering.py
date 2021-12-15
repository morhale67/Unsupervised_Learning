import sklearn.cluster as cls


def clustering(method, x_data, k=5):
    if method == 'Kmeans':
        clusters = cls.KMeans(k, random_state=0).fit(x_data)
    elif method == 'Dbscan':
        clusters = cls.DBSCAN(eps=3, min_samples=30).fit(x_data)
    elif method == 'Agglomerative':
        clusters = cls.AgglomerativeClustering(linkage='average', n_clusters=10).fit(x_data)
    else:
        print("try again")
        clusters = []
    return clusters