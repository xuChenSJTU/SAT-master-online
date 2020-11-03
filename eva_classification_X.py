import os
import matplotlib.pyplot as plt
from evaluation import class_eva
import pickle
from sklearn.utils import shuffle
import numpy as np
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold

'''
#####################################
This is the script for evaluating the classification performance for only generated X
The involved datasets are: cora, citeseer, pubmed, ms_academic
The involved methods are: # NeighAggre, GCN, VAE, LFI and MLP
'''


'''
calculate the accuracy
'''
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# set necessary parameters
method_name = 'LFI'  # NeighAggre, GCN, VAE, LFI, GAT
dataset = 'cora'  # cora, citeseer, pubmed
train_fts_ratio = 0.4*1.0
c = 10.0
print('begining......')

# load necessary data
if method_name=='LFI':
    gene_fts = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}_G1.0_R1.0_C{}.pkl'.format(dataset, train_fts_ratio, c)), 'rb'))
else:
    gene_fts = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                             'gene_fts_train_ratio_{}_{}.pkl'.format(dataset, train_fts_ratio)), 'rb'))

gene_fts_idx = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_{}_test_fts_idx.pkl'.format(dataset, train_fts_ratio)), 'rb'))
true_features = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_true_features.pkl'.format(dataset)), 'rb'))

all_labels = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset,
                                            '{}_labels.pkl'.format(dataset)), 'rb'))

gene_fts = gene_fts[gene_fts_idx]
#
# gene_fts = true_features[gene_fts_idx]


labels_of_gene = all_labels[gene_fts_idx]


# make classification evaluation process
gene_data = np.concatenate((gene_fts, np.reshape(labels_of_gene, newshape=[-1, 1])), axis=1)

final_list = []
for i in range(10):
    gene_data = shuffle(gene_data, random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(gene_data)
    acc_list = []
    for train_idx, test_idx in split_data:
        train_data = gene_data[train_idx]

        train_featured_idx = np.where(train_data.sum(1)!=0)[0]
        train_data = train_data[train_featured_idx]

        test_data = gene_data[test_idx]

        acc = class_eva(train_fts=train_data[:, :-1], train_lbls=train_data[:, -1],
                      test_fts=test_data[:, :-1], test_lbls=test_data[:, -1])
        acc_list.append(acc)
    avg_acc = np.mean(acc_list)
    final_list.append(avg_acc)

print('dataset: {}, method: {}, ratio: {}, lambda_c: {}'.format(dataset, method_name, train_fts_ratio, c))
print('classification performance: {}'.format(np.mean(final_list)))




# '''
# plot the accuracy figure
# '''
# # set necessary parameters
# dataset = 'cora'  # cora, citeseer, pubmed, reddit
#
# method_name_list = ['NeighAggre', 'GCN', 'VAE', 'LFI']  # NeighAggre, GCN, VAE, LFI
#
# train_fts_ratio = 0.4
#
# acc_dict = {}
#
# # plot the results
# for name in method_name_list:
#     acc_list = pickle.load(open(os.path.join(os.getcwd(), 'features', name, '{}_acc_list.pkl').format(dataset), 'rb'))
#     acc_dict[name] = acc_list
#
# # plot loss curve
# font = {'family': 'Times New Roman',
#         'color': 'black',
#         'weight': 'bold',
#         'size': 15,
#         }
# mycolor = np.array([[224, 32, 32],
#                         [255, 192, 0],
#                         [32, 160, 64],
#                         [48, 96, 192],
#                         [192, 48, 192]]) / 255.0
# mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']
#
# my_line_width = 3
# my_marker_size = 10
#
# # plot train G/D curve
# plt.figure(1)
# plt.style.use('ggplot')
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
# x_axix = range(5)
# plt.plot(x_axix, acc_dict['NeighAggre'], color=mycolor[0], label='NeighAggre', linewidth=my_line_width,
#              markersize=my_marker_size)
# plt.plot(x_axix, acc_dict['GCN'], color=mycolor[1], label='GCN', linewidth=my_line_width,
#              markersize=my_marker_size)
# plt.plot(x_axix, acc_dict['VAE'], color=mycolor[2], label='VAE', linewidth=my_line_width,
#              markersize=my_marker_size)
# plt.plot(x_axix, acc_dict['LFI'], color=mycolor[3], label='LFI', linewidth=my_line_width,
#              markersize=my_marker_size)
#
# my_legend = plt.legend(loc='upper right', fontsize=15)
# frame = my_legend.get_frame()
# frame.set_alpha(1)
# frame.set_facecolor('none')
#
# plt.tick_params(labelsize='10')
# plt.xlabel('Add ratio', fontdict=font)
# plt.ylabel('Accuracy', fontdict=font)
# # plt.show()
# plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_{}_acc_comparison.png'.format(dataset, train_fts_ratio)))
#
#
#
