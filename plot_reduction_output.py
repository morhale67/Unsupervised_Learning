import matplotlib.pyplot as plt
import os


def plot1_reduction_output(converted_data, y_data, x_label, y_label, method_name):
    """x-y graph"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    c_map = plt.cm.get_cmap('jet', 10)
    plt.scatter(converted_data[:, 0], converted_data[:, 1], s=15,
                cmap=c_map, c=y_data)
    plt.colorbar()
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.title('Reducion by ' + method_name)

    # my_path = os.path.abspath(r'C:\Users\user\Desktop\University\Unsupervised_learning\figures')  # Figures out the absolute path for you in case your working directory moves around.
    # my_file = method_name
    # plt.savefig(os.path.join(my_path, my_file))


def multiplot_reduction_output(all_x, all_y, method_name):
    c_map = plt.cm.get_cmap('jet', 10)

    ax = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2, xticks=[], yticks=[])
    ax1.scatter(all_x[0][:, 0], all_x[0][:, 1], s=15, cmap=c_map, c=all_y[0])
    ax1.set_title('reduction by PCA', fontsize=24)

    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, xticks=[], yticks=[])
    ax2.scatter(all_x[1][:, 0], all_x[1][:, 1], s=15, cmap=c_map, c=all_y[1])
    ax2.set_title('reduction by CMD', fontsize=24)

    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2, xticks=[], yticks=[])
    ax3.scatter(all_x[2][:, 0], all_x[2][:, 1], s=15, cmap=c_map, c=all_y[2])
    ax3.set_title('reduction by Isomap', fontsize=24)

    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2, xticks=[], yticks=[])
    ax4.scatter(all_x[3][:, 0], all_x[3][:, 1], s=15, cmap=c_map, c=all_y[3])
    ax4.set_title('reduction by Laplacian', fontsize=24)

    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2, xticks=[], yticks=[])
    ax5.scatter(all_x[4][:, 0], all_x[4][:, 1], s=15, cmap=c_map, c=all_y[4])
    ax5.set_title('reduction by LLE', fontsize=24)

    ax.suptitle(method_name + ' Algorithem', fontsize=34)
    plt.show()
