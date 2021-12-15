from reduce_dimension import reduce_dimension
from clustering import clustering


def demo(clustering_method, x_data, y_data):
    all_y_cls, all_reduced_x = [], []
    all_methods = ['pca', 'cmds', 'isomap', 'laplacian', 'lle']
    for reduce_method in all_methods:
        cur_reduced_x = reduce_dimension(reduce_method, x_data)
        clusters = clustering(clustering_method, cur_reduced_x, 9)
        cur_y_cls = clusters.labels_
        all_reduced_x.append(cur_reduced_x)
        all_y_cls.append(cur_y_cls)
    return all_reduced_x, all_y_cls

