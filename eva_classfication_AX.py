import os
import matplotlib.pyplot as plt
from evaluation import class_eva
import pickle
from sklearn.utils import shuffle
import numpy as np
import scipy.sparse as sp
from GCN_models import GCN_eva
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from utils import normalize_adj
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold

'''
#####################################
This is the script for evaluating the classification performance with only A used and A+X used
The involved datasets are: cora, citeseer, pubmed, ms_academic
The involved methods are: DeepWalk, GCN and MLP
'''


# '''
# calculate the accuracy for only A used
# '''
# # os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# # set necessary parameters
# method_name = 'GCN'  # DeepWalk, GCN
# dataset = 'ms_academic'  # cora, citeseer, pubmed, ms_academic
# train_fts_ratio = 0.4
#
#
# is_cuda = torch.cuda.is_available()
# # load necessary data
# adj = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
#                                             '{}_sp_adj.pkl'.format(dataset)), 'rb'))
#
# gene_fts_idx = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
#                                             '{}_{}_test_fts_idx.pkl'.format(dataset, train_fts_ratio)), 'rb'))
#
# all_labels = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset,
#                                             '{}_labels.pkl'.format(dataset)), 'rb'))
#
#
# adj = adj[gene_fts_idx, :][:, gene_fts_idx]
# n_nodes = adj.shape[0]
# indices = np.where(adj != 0)
# rows = indices[0]
# cols = indices[1]
# adj = sp.coo_matrix((np.ones(shape=len(rows)), (rows, cols)), shape=[n_nodes, n_nodes])
#
# adj = normalize_adj(adj + sp.eye(adj.shape[0]))
# indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
# values = torch.FloatTensor(adj.tocoo().data)
#
# adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
#
# labels_of_gene = torch.LongTensor(all_labels[gene_fts_idx])
# n_class = max(labels_of_gene).item()+1
#
#
# # generate ont-hot features for all nodes, this means no node feature is used
# indices = torch.LongTensor(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0))
# values = torch.FloatTensor(np.ones(indices.shape[1]))
# features = torch.sparse.FloatTensor(indices, values, torch.Size([n_nodes, n_nodes]))
#
# final_list = []
# for i in range(10):
#     node_Idx = shuffle(np.arange(labels_of_gene.shape[0]), random_state=72)
#     KF = KFold(n_splits=5)
#     split_data = KF.split(node_Idx)
#     acc_list = []
#     for train_idx, test_idx in split_data:
#         train_idx = torch.LongTensor(train_idx)
#         test_idx = torch.LongTensor(test_idx)
#         train_lbls = labels_of_gene[train_idx]
#         test_lbls = labels_of_gene[test_idx]
#
#         test_lbls_arr = test_lbls.numpy()
#
#         model = GCN_eva(nfeat=n_nodes, nhid=64, nclass=n_class, dropout=0.5)
#         if is_cuda:
#             model.cuda()
#             adj = adj.cuda()
#             features = features.cuda()
#             train_lbls = train_lbls.cuda()
#             test_lbls = test_lbls.cuda()
#             train_idx = train_idx.cuda()
#             test_idx = test_idx.cuda()
#
#         lossfunc = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
#
#         best_acc = 0
#         for epoch in range(1000):
#             model.train()
#
#             optimizer.zero_grad()
#             output = model(features, adj)
#
#             loss_train = F.nll_loss(output[train_idx], train_lbls)
#             loss_train.backward()
#             optimizer.step()
#
#
#             model.eval()
#             val_loss = F.nll_loss(output[test_idx], test_lbls)
#             if is_cuda:
#                 preds = np.argmax(output[test_idx].data.cpu().numpy(), axis=1)
#             else:
#                 preds = np.argmax(output[test_idx].data.numpy(), axis=1)
#
#             acc = np.sum(preds == test_lbls_arr)*1.0/len(preds)
#             if acc > best_acc:
#                 best_acc = acc
#             print('Round:{}, Epoch: {}, train loss: {:.4f}, vali loss: {:.4f}, acc: {}'.format(i, epoch, loss_train.item(), val_loss.item(), acc))
#
#         acc_list.append(best_acc)
#     avg_acc = np.mean(acc_list)
#     final_list.append(avg_acc)
# print('GCN, dataset; {}, avg accuracy:{}'.format(dataset, np.mean(final_list)))
#

'''                                                                                                                                                   
calculate the accuracy for only A+X used                                                                                                                
'''
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# set necessary parameters
method_name = 'LFI'  # NeighAggre, GCN, VAE, LFI, GAT
dataset = 'cora'  # cora, citeseer, pubmed
train_fts_ratio = 0.4*1.0
c = 10.0
print('begining......')

is_cuda = torch.cuda.is_available()
# load necessary data
true_features = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_true_features.pkl'.format(dataset)), 'rb'))

if method_name=='LFI':
    gene_fts = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}_G1.0_R1.0_C{}.pkl'.format(dataset, train_fts_ratio, c)), 'rb'))
else:
    gene_fts = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                             'gene_fts_train_ratio_{}_{}.pkl'.format(dataset, train_fts_ratio)), 'rb'))

adj = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_sp_adj.pkl'.format(dataset)), 'rb'))

gene_fts_idx = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_{}_test_fts_idx.pkl'.format(dataset, train_fts_ratio)), 'rb'))

all_labels = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset,
                                            '{}_labels.pkl'.format(dataset)), 'rb'))


adj = adj[gene_fts_idx, :][:, gene_fts_idx]
n_nodes = adj.shape[0]
indices = np.where(adj != 0)
rows = indices[0]
cols = indices[1]
adj = sp.coo_matrix((np.ones(shape=len(rows)), (rows, cols)), shape=[n_nodes, n_nodes])

adj = normalize_adj(adj + sp.eye(adj.shape[0]))
indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
values = torch.FloatTensor(adj.tocoo().data)

adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

labels_of_gene = torch.LongTensor(all_labels[gene_fts_idx])
n_class = max(labels_of_gene).item()+1

# make choice of whether generated fts or true fts
features = torch.FloatTensor(gene_fts[gene_fts_idx])
# features = torch.FloatTensor(true_features[gene_fts_idx])


final_list = []
for i in range(10):
    node_Idx = shuffle(np.arange(labels_of_gene.shape[0]), random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(node_Idx)
    acc_list = []
    for train_idx, test_idx in split_data:
        train_idx = torch.LongTensor(train_idx)
        test_idx = torch.LongTensor(test_idx)

        train_fts = features[train_idx]
        test_fts = features[test_idx]

        featured_train_idx = train_idx[(train_fts.sum(1)!=0).nonzero().reshape([-1])]
        featured_test_idx = test_idx[(test_fts.sum(1)!=0).nonzero().reshape([-1])]
        non_featured_test_idx = test_idx[(test_fts.sum(1)==0).nonzero().reshape([-1])]

        featured_train_lbls = labels_of_gene[featured_train_idx]
        featured_test_lbls = labels_of_gene[featured_test_idx]
        non_featured_test_lbls = labels_of_gene[non_featured_test_idx]

        featured_test_lbls_arr = featured_test_lbls.numpy()
        non_featured_test_lbls_arr = non_featured_test_lbls.numpy()

        model = GCN_eva(nfeat=features.shape[1], nhid=64, nclass=n_class, dropout=0.5, input_fts_sparse=False)
        if is_cuda:
            model.cuda()
            adj = adj.cuda()
            features = features.cuda()
            featured_train_lbls = featured_train_lbls.cuda()
            featured_test_lbls = featured_test_lbls.cuda()
            featured_train_idx = featured_train_idx.cuda()
            featured_test_idx = featured_test_idx.cuda()


        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

        best_acc = 0
        for epoch in range(1000):
            model.train()

            optimizer.zero_grad()
            output = model(features, adj)

            loss_train = F.nll_loss(output[featured_train_idx], featured_train_lbls)
            loss_train.backward()
            optimizer.step()


            model.eval()
            val_loss = F.nll_loss(output[featured_test_idx], featured_test_lbls)
            if is_cuda:
                featured_preds = np.argmax(output[featured_test_idx].data.cpu().numpy(), axis=1)
            else:
                featured_preds = np.argmax(output[featured_test_idx].data.numpy(), axis=1)

            random_preds = np.random.choice(np.arange(n_class), len(non_featured_test_idx))

            preds = np.concatenate((featured_preds, random_preds))
            lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls_arr))

            acc = np.sum(preds == lbls)*1.0/len(preds)
            if acc > best_acc:
                best_acc = acc
            print('Round:{}, Epoch: {}, train loss: {:.4f}, vali loss: {:.4f}, acc: {}'.format(i, epoch, loss_train.item(), val_loss.item(), acc))

        acc_list.append(best_acc)
    avg_acc = np.mean(acc_list)
    final_list.append(avg_acc)
print('GCN(A+X), dataset; {}, X of method: {}, avg accuracy:{}, ratio: {}, lambda_c: {}'.format(dataset,
                                                                                                method_name,
                                                                                                np.mean(final_list),
                                                                                                train_fts_ratio, c))
