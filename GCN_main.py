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
from sklearn.utils import shuffle
from utils import load_data, accuracy, new_load_data, load_generated_features
from GCN_models import GCN, GAT
import pickle
from evaluation import RECALL_NDCG, CAL_BCE

# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
use_feature = False

method_name = 'GAT' # GAT or GCN
train_fts_ratio = 0.4*1.0
topK_list = [3, 5, 10]

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--dataset', type=str, default='pubmed', help='cora, citeseer, steam, pubmed')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print('loading dataset: {}'.format(args.dataset))
adj, true_features, _, _, _, _ = new_load_data(args.dataset, norm_adj=True, generative_flag=True)

# generate ont-hot features for all nodes, this means no node feature is used
indices = torch.LongTensor(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0))
values = torch.FloatTensor(np.ones(indices.shape[1]))
features = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))

# split train features and generative features
shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=args.seed)
train_fts_idx = torch.LongTensor(shuffled_nodes[:int(train_fts_ratio * adj.shape[0])])
vali_fts_idx = torch.LongTensor(shuffled_nodes[int(0.4 * adj.shape[0]):int((0.4 + 0.1) * adj.shape[0])])
test_fts_idx = torch.LongTensor(shuffled_nodes[int((0.4 + 0.1) * adj.shape[0]):])

# pickle.dump(train_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_train_fts_idx.pkl'.format(
#     args.dataset, train_fts_ratio)), 'wb'))
# pickle.dump(vali_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_vali_fts_idx.pkl'.format(
#     args.dataset, train_fts_ratio)), 'wb'))
# pickle.dump(test_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_test_fts_idx.pkl'.format(
#     args.dataset, train_fts_ratio)), 'wb'))

if method_name=='GCN':
    model = GCN(nfeat=adj.shape[1],
                    nhid=args.hidden,
                    nclass=true_features.shape[1],
                    dropout=args.dropout)
elif method_name=='GAT':
    model = GAT(nfeat=adj.shape[1],
                    nhid=args.hidden,
                    nclass=true_features.shape[1],
                    dropout=args.dropout, alpha=args.alpha, nheads=args.nb_heads)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    true_features = true_features.cuda()
    train_fts_idx = train_fts_idx.cuda()
    vali_fts_idx = vali_fts_idx.cuda()
    test_fts_idx = test_fts_idx.cuda()

diag_fts, adj, train_fts, val_fts = Variable(features), Variable(adj), Variable(true_features[train_fts_idx]), Variable(true_features[vali_fts_idx])
test_fts = Variable(true_features[test_fts_idx])

def compute_test():
    model.eval()
    output_fts = model(diag_fts, adj)
    loss_test = loss_function(output_fts[test_fts_idx], test_fts, pos_weight_tensor, neg_weight_tensor)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))

    if args.dataset in ['cora', 'citeseer', 'steam']:
        output_fts = torch.sigmoid(output)
    elif args.dataset in ['pubmed']:
        output_fts = output

    if args.cuda:
        train_fts_arr = true_features[train_fts_idx].data.cpu().numpy()
        vali_fts_arr = true_features[vali_fts_idx].data.cpu().numpy()
        test_fts_arr = output_fts[test_fts_idx].data.cpu().numpy()
        train_fts_idx_arr = train_fts_idx.data.cpu().numpy()
        vali_fts_idx_arr = vali_fts_idx.data.cpu().numpy()
        test_fts_idx_arr = test_fts_idx.data.cpu().numpy()
    else:
        train_fts_arr = true_features[train_fts_idx].data.numpy()
        vali_fts_arr = true_features[vali_fts_idx].data.numpy()
        test_fts_arr = output_fts[test_fts_idx].data.numpy()
        train_fts_idx_arr = train_fts_idx.data.numpy()
        vali_fts_idx_arr = vali_fts_idx.data.numpy()
        test_fts_idx_arr = test_fts_idx.data.numpy()

    save_fts = np.zeros(shape=true_features.shape)
    save_fts[train_fts_idx_arr] = train_fts_arr
    save_fts[vali_fts_idx_arr] = vali_fts_arr
    save_fts[test_fts_idx_arr] = test_fts_arr

    print('Saving generated features and true features......')
    pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))

# set loss instances from classes
BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE = torch.nn.MSELoss(reduction='none')

def loss_function_discrete(output_fts, fts_labls, pos_weight_tensor, neg_weight_tensor):
    output_fts_reshape = torch.reshape(output_fts, shape=[-1])
    fts_labls_reshape = torch.reshape(fts_labls, shape=[-1])
    weight_mask = torch.where(fts_labls_reshape != 0.0, pos_weight_tensor, neg_weight_tensor)
    loss_bce = torch.mean(BCE(output_fts_reshape, fts_labls_reshape) * weight_mask)
    return loss_bce

def loss_function_continuous(output_fts, fts_labls, pos_weight_tensor, neg_weight_tensor):
    output_fts_reshape = torch.reshape(output_fts, shape=[-1])
    fts_labls_reshape = torch.reshape(fts_labls, shape=[-1])
    loss_mse = torch.mean(MSE(output_fts_reshape, fts_labls_reshape))
    return loss_mse


 # set loss function and pos weight
if args.dataset in ['cora', 'citeseer', 'steam']:
    loss_function = loss_function_discrete
    pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item() / (torch.sum(true_features[train_fts_idx] != 0.0).item())
elif args.dataset in ['reddit', 'pinterest', 'wikipedia', 'pubmed', 'ms_academic']:
    loss_function = loss_function_continuous
    pos_weight = 1.0

if args.cuda:
    pos_weight_tensor = torch.FloatTensor([pos_weight]).cuda()
    neg_weight_tensor = torch.FloatTensor([1.0]).cuda()
else:
    pos_weight_tensor = torch.FloatTensor([pos_weight])
    neg_weight_tensor = torch.FloatTensor([1.0])

# Train model
t_total = time.time()
loss_values = []
eva_values_list = []
bad_counter = 0
best_epoch = 0
best_mse = 1000.0
best_recall = 0.0
for epoch in range(args.epochs):
    t = time.time()
    model.train()

    optimizer.zero_grad()
    output = model(diag_fts, adj)


    loss_train = loss_function(output[train_fts_idx], train_fts, pos_weight_tensor, neg_weight_tensor)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(diag_fts, adj)

    loss_val = loss_function(output[vali_fts_idx], val_fts, pos_weight_tensor, neg_weight_tensor)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.8f}'.format(loss_train.item()),
          'loss_val: {:.8f}'.format(loss_val.item()))

    loss_val_value = loss_val.item()
    loss_values.append(loss_val_value)

    '''
    make early stop condition
    '''
    if args.dataset in ['cora', 'citeseer', 'steam']:
        if args.cuda:
            gene_fts = output[vali_fts_idx].data.cpu().numpy()
            gt_fts = true_features[vali_fts_idx].cpu().numpy()
        else:
            gene_fts = output[vali_fts_idx].data.numpy()
            gt_fts = true_features[vali_fts_idx].numpy()

        avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=topK_list[0])
        eva_values_list.append(avg_recall)
        if eva_values_list[-1] > best_recall:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                        'best_gcn_{}_{}.pkl'.format(args.dataset, train_fts_ratio)))
            best_recall = eva_values_list[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        # if bad_counter == args.patience:
        #     break

    elif args.dataset in ['pubmed']:
        eva_values_list.append(loss_val_value)

        if eva_values_list[-1] < best_mse:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                        'best_gcn_{}_{}.pkl'.format(args.dataset, train_fts_ratio)))
            best_mse = eva_values_list[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'output', method_name, 'best_gcn_{}_{}.pkl'.format(args.dataset, train_fts_ratio))))

# Testing and save generated fts
compute_test()

# find neighbors and make raw feature aggregation for unknown nodes
model.eval()
output_fts = model(diag_fts, adj)

if args.dataset in ['cora', 'citeseer', 'steam']:
    loss_test = loss_function(output_fts[test_fts_idx], test_fts, pos_weight_tensor, neg_weight_tensor)
    print('BCE loss: {}'.format(loss_test.item()))

# pickle.dump(model.z, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_latent_Z.pkl'.format(
#     args.dataset, train_fts_ratio)), 'wb'))

print('test for label propagation......')
if args.cuda:
    gene_test_fts = output_fts[test_fts_idx].data.cpu().numpy()
    gt_fts = true_features[test_fts_idx].cpu().numpy()
else:
    gene_test_fts = output_fts[test_fts_idx].data.numpy()
    gt_fts = true_features[test_fts_idx].numpy()

if args.dataset in ['cora', 'citeseer', 'steam']:
    '''
    evaluation for Recall and NDCG
    '''
    for topK in topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_test_fts, gt_fts, topN=topK)
        print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
    print('method: {}, dataset: {}'.format(method_name, args.dataset))
elif args.dataset in ['pubmed']:
    NL2 = np.mean(np.linalg.norm(gene_test_fts - gt_fts, axis=1) / np.linalg.norm(gt_fts, axis=1))
    print('normalized L2 distance: {:.8f}'.format(NL2))


'''
save necessary fts for evaluation for continuous fts
'''

known_node_idx = torch.cat([train_fts_idx, vali_fts_idx])
unknown_node_idx = test_fts_idx

if args.cuda:
    known_node_idx = known_node_idx.cpu().data.numpy()
    unknown_node_idx = unknown_node_idx.cpu().data.numpy()
    true_features = true_features.cpu().data.numpy()
else:
    known_node_idx = known_node_idx.data.numpy()
    unknown_node_idx = unknown_node_idx.data.numpy()
    true_features = true_features.data.numpy()


# pickle.dump(known_node_idx, open(os.path.join(os.getcwd(), 'features', method_name,
#                                             'known_idx_train_ratio_{}_{}.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))
# pickle.dump(unknown_node_idx, open(os.path.join(os.getcwd(), 'features', method_name,
#                                             'unknown_idx_train_ratio_{}_{}.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))
# pickle.dump(true_features, open(os.path.join(os.getcwd(), 'features', method_name,
#                                             'true_features_{}.pkl'.format(args.dataset)), 'wb'))
print('method: {}, dataset: {}, hidden: {}, ratio: {}'.format(method_name, args.dataset, args.hidden, train_fts_ratio))
