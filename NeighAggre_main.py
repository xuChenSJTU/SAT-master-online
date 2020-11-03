from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
from utils import load_data, accuracy, new_load_data
from GCN_models import GCN
from sklearn.utils import shuffle
from evaluation import RECALL_NDCG

os.environ['CUDA_VISIBLE_DEVICES'] = ' '
method_name = 'NeighAggre'
train_fts_ratio = 0.4*0.2
topK_list = [10, 20, 50]

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='pubmed', help='cora, citeseer, steam')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print('loading dataset: {}'.format(args.dataset))
# note that the node_class_lbls, node_idx_train, node_idx_val, node_idx_test are only used for evaluation.
adj, true_features, node_class_lbls, _, _, _ = new_load_data(args.dataset, norm_adj=False, generative_flag=True)
adj = adj.to_dense()
# pickle.dump(adj.numpy(), open(os.path.join(os.getcwd(), 'features', 'NeighAggre',
#                                             '{}_sp_adj.pkl'.format(args.dataset)), 'wb'))
#
# pickle.dump(true_features.data.numpy(), open(os.path.join(os.getcwd(), 'features', 'NeighAggre',
#                                             '{}_true_features.pkl'.format(args.dataset)), 'wb'))

# generate ont-hot features for all nodes, this means no node feature is used
indices = torch.LongTensor(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0))
values = torch.FloatTensor(np.ones(indices.shape[1]))
features = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))

# split train features and generative features
shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=args.seed)
train_fts_idx = torch.LongTensor(shuffled_nodes[:int(train_fts_ratio*adj.shape[0])])
vali_fts_idx = torch.LongTensor(shuffled_nodes[int(0.4*adj.shape[0]):int((0.4+0.1)*adj.shape[0])])
test_fts_idx = torch.LongTensor(shuffled_nodes[int((0.4+0.1)*adj.shape[0]):])

pickle.dump(train_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_train_fts_idx.pkl'.format(
    args.dataset, train_fts_ratio)), 'wb'))
pickle.dump(vali_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_vali_fts_idx.pkl'.format(
    args.dataset, train_fts_ratio)), 'wb'))
pickle.dump(test_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_test_fts_idx.pkl'.format(
    args.dataset, train_fts_ratio)), 'wb'))

# find neighbors and make raw feature aggregation for unknown nodes
# since we only have the train node fts to aggregate and normalize with mean operation, so we set zero to nodes without fts.
mask_adj = torch.zeros_like(adj)
mask_adj = mask_adj[test_fts_idx, :]
mask_adj[:, train_fts_idx] = adj[test_fts_idx, :][:, train_fts_idx]

aggregation_fts = torch.mm(mask_adj, true_features)/torch.reshape(mask_adj.sum(1)+1e-24, shape=[-1, 1])

save_fts = torch.zeros_like(true_features)
save_fts[test_fts_idx] = aggregation_fts

print('Saving generated features and true features......')
pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))

print('test for label propagation......')
if args.cuda:
    aggregation_fts = aggregation_fts.cpu().numpy()
    gt_fts = true_features[test_fts_idx].cpu().numpy()
else:
    aggregation_fts = aggregation_fts.numpy()
    gt_fts = true_features[test_fts_idx].numpy()


if args.dataset in ['cora', 'citeseer', 'steam']:
    for topK in topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(aggregation_fts, gt_fts, topN=topK)
        print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))

elif args.dataset in ['pubmed']:
    NL2 = np.mean(np.linalg.norm(aggregation_fts - gt_fts, axis=1)/(np.linalg.norm(gt_fts, axis=1)))
    print('normalized L2 distance: {:.8f}'.format(NL2))
print('method: {}, dataset: {}, ratio: {}'.format(method_name, args.dataset, train_fts_ratio))
