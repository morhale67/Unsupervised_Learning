from load_data import load_data
from Demo import demo
from plot_reduction_output import multiplot_reduction_output
from plot_reduction_output import plot1_reduction_output


x_data, y_data = load_data()

all_reduced_x, all_y_cls = demo('Dbscan', x_data, y_data)
multiplot_reduction_output(all_reduced_x, all_y_cls, method_name='DBSCAN')




