from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding


def reduce_dimension(method, x_data):
    if method == 'pca':
        converted_data = pca_reduction(x_data)
    elif method == 'laplacian':
        converted_data = laplacian_reduction(x_data)
    elif method == 'lle':
        converted_data = lle_reduction(x_data)
    elif method == 'cmds':
        converted_data = cmds_reduction(x_data)
    elif method == 'isomap':
        converted_data = isomap_reduction(x_data)
    else:
        print("try again")
        converted_data = []
    return converted_data


def pca_reduction(x_data):
    pca = PCA(2)  # we need 2 principal components.
    converted_data = pca.fit_transform(x_data)
    return converted_data


def laplacian_reduction(x_data):
    embedding = SpectralEmbedding(n_components=2)
    converted_data = embedding.fit_transform(x_data)
    return converted_data


def lle_reduction(x_data, n_neighbors=12):
    embedding = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, method="modified")
    converted_data = embedding.fit_transform(x_data)
    return converted_data


def cmds_reduction(x_data):
    dist_metric = euclidean_distances(x_data)
    mds = MDS(metric=True, dissimilarity='precomputed', random_state=0)  # we need 2 principal components.
    converted_data = mds.fit_transform(dist_metric)
    return converted_data


def isomap_reduction(x_data, n_neighbors=5):
    embedding = Isomap(n_components=2, n_neighbors=n_neighbors)
    converted_data = embedding.fit_transform(x_data)
    return converted_data



