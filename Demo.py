from reduce_dimension import reduce_dimension
from clustering import clustering
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans


def demo(clustering_method, x_data, y_data):
    all_y_cls, all_reduced_x, sil_score = [], [], []
    all_reduce_methods = ['tsne', 'ica', 'pca', 'cmds', 'isomap', 'laplacian', 'lle']
    for reduce_method in all_reduce_methods:
        cur_reduced_x = reduce_dimension(reduce_method, x_data)
        clusters = clustering(clustering_method, cur_reduced_x, 9)
        cur_y_cls = clusters.labels_
        all_reduced_x.append(cur_reduced_x)
        all_y_cls.append(cur_y_cls)

        sil_score.append(silhouette_score(cur_reduced_x, cur_y_cls))

        # model = SilhouetteVisualizer(KMeans(5))
        # model.fit(cur_reduced_x)
        # model.show()

    return all_reduced_x, all_y_cls, sil_score

