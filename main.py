from load_data import load_data
from Demo import demo
from plot_reduction_output import multiplot_reduction_output

all_sil_score = []
x_data, y_data = load_data()

all_clustering_method = ['Kmeans', 'Dbscan', 'Agglomerative']
for clustering_method in all_clustering_method:
    all_reduced_x, all_y_cls, cur_sil_score = demo(clustering_method, x_data, y_data)
    multiplot_reduction_output(all_reduced_x, all_y_cls, clustering_method)
    all_sil_score.append(cur_sil_score)

print('end')

