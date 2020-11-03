import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
import pickle as pkl
import torch.nn.functional as F
import networkx as nx
import os
from sklearn.svm import SVC
from sklearn.utils import shuffle

def CAL_BCE(estimated_fts, true_fts):
    estimated_fts = torch.FloatTensor(estimated_fts)
    true_fts = torch.FloatTensor(true_fts)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        estimated_fts = estimated_fts.cuda()
        true_fts = true_fts.cuda()
    ce_loss = F.binary_cross_entropy(estimated_fts.reshape([-1]), true_fts.reshape([-1])).item()
    return ce_loss

def RECALL_NDCG(estimated_fts, true_fts, topN = 10):
    preds = np.argsort(-estimated_fts, axis=1)
    preds = preds[:, :topN]

    gt = [np.where(true_fts[i, :]!=0)[0] for i in range(true_fts.shape[0])]
    recall_list = []
    ndcg_list = []
    for i in range(preds.shape[0]):
        # calculate recall
        if len(gt[i]) != 0:

            # whether the generated feature is non feature
            if np.sum(estimated_fts[i,:])!=0:
                recall = len(set(preds[i, :]) & set(gt[i]))*1.0/len(set(gt[i]))
                recall_list.append(recall)

                # calculate ndcg
                intersec = np.array(list(set(preds[i, :]) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(preds[i, :] == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0/(np.log2(x+1+1)) for x in dcg])
                    idcg = np.sum([1.0/(np.log2(x+1+1)) for x in range(len(gt[i]))])
                    ndcg = dcg*1.0/idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)
            else:
                temp_preds = shuffle(np.arange(estimated_fts.shape[1]))[:topN]

                recall = len(set(temp_preds) & set(gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                # calculate ndcg
                intersec = np.array(list(set(temp_preds) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(temp_preds == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    return avg_recall, avg_ndcg


class MLP(nn.Module):
    def __init__(self, fts_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 =nn.Linear(fts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_fts):
        h1 = F.relu(self.fc1(input_fts))
        h2 = self.fc2(h1)
        return F.log_softmax(h2, dim=1)

def class_eva(train_fts, train_lbls, test_fts, test_lbls):

    test_featured_idx = np.where(test_fts.sum(1)!=0)[0]
    test_non_featured_idx = np.where(test_fts.sum(1)==0)[0]

    featured_test_fts = test_fts[test_featured_idx]
    featured_test_lbls = test_lbls[test_featured_idx]
    non_featured_test_lbls = test_lbls[test_non_featured_idx]

    fts_dim = train_fts.shape[1]
    hid_dim = 64
    n_class = int(max(max(train_lbls), max(test_lbls)) + 1)
    is_cuda = torch.cuda.is_available()

    model = MLP(fts_dim, hid_dim, n_class)
    if is_cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    featured_test_lbls_arr = featured_test_lbls.copy()

    train_fts = torch.FloatTensor(train_fts)
    train_lbls = torch.LongTensor(train_lbls)
    featured_test_fts = torch.FloatTensor(featured_test_fts)
    featured_test_lbls = torch.LongTensor(featured_test_lbls)
    if is_cuda:
        train_fts = train_fts.cuda()
        train_lbls = train_lbls.cuda()
        featured_test_fts = featured_test_fts.cuda()
        featured_test_lbls = featured_test_lbls.cuda()

    acc_list = []
    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_fts)

        loss = F.nll_loss(outputs, train_lbls)
        loss.backward()
        optimizer.step()

        # make evaluation process
        model.eval()
        featured_test_outputs = model(featured_test_fts)
        test_loss = F.nll_loss(featured_test_outputs, featured_test_lbls)
        if is_cuda:
            featured_test_outputs = featured_test_outputs.data.cpu().numpy()
        else:
            featured_test_outputs = featured_test_outputs.data.numpy()
        featured_preds = np.argmax(featured_test_outputs, axis=1)

        random_preds = np.random.choice(n_class, len(test_non_featured_idx))

        preds = np.concatenate((featured_preds, random_preds))
        lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls))

        acc = np.sum(preds==lbls)*1.0/len(lbls)
        acc_list.append(acc)
        print('Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(i, loss.item(), test_loss.item(), acc))

    print('Best epoch:{}, best acc: {:.4f}'.format(np.argmax(acc_list), np.max(acc_list)))
    return np.max(acc_list)
