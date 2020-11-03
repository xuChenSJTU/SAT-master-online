import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def smooth(MMD_list, n_smooth=4):
    res_list = []
    for i in range(len(MMD_list)):
        if MMD_list[i] > 10.0:
            res_list.append(res_list[-1])
        else:
            res_list.append(MMD_list[i])
    final_list = []
    for i in range(0, len(res_list)-n_smooth):
        final_list.append(np.mean(res_list[i:i+n_smooth]))
    return final_list

dataset_file = 'cora'
train_fts_ratio = 0.4
plot_choice = 'self'  # self
n_smooth = 1



if plot_choice == 'self':
    MMD_list_joint_train = pickle.load(
        open(os.path.join(os.getcwd(), 'features', 'LFI', '{}_train_MMD_list_G1.0_C10.0_R1.0.pkl'.format(dataset_file)), 'rb'))
    MMD_list_joint_vali = pickle.load(
        open(os.path.join(os.getcwd(), 'features', 'LFI', '{}_vali_MMD_list_G1.0_C10.0_R1.0.pkl'.format(dataset_file)), 'rb'))


    MMD_list_joint_train = smooth(MMD_list_joint_train, n_smooth=n_smooth)
    MMD_list_joint_vali = smooth(MMD_list_joint_vali, n_smooth=n_smooth)

    # plot loss curve
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 15,
            }
    mycolor = np.array([[224, 32, 32],
                        [255, 192, 0],
                        [32, 160, 64],
                        [48, 96, 192],
                        [192, 48, 192]]) / 255.0
    mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

    my_line_width = 3
    my_marker_size = 10

    # plot train G/D curve
    plt.figure(1)
    plt.style.use('ggplot')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    x_axix = range(len(MMD_list_joint_train))
    plt.plot(x_axix, MMD_list_joint_train, color=mycolor[0], label='Train', linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, MMD_list_joint_vali, color=mycolor[1], label='Vali', linewidth=my_line_width,
             markersize=my_marker_size)

    my_legend = plt.legend(loc='upper right', fontsize=15)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.tick_params(labelsize='10')
    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('MMD Distance', fontdict=font)
    # plt.show()
    plt.savefig(
        os.path.join(os.getcwd(), 'figures', 'LFI', '{}_{}_self_train_MMD.png'.format(dataset_file, train_fts_ratio)))
