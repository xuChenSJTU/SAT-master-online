from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from VAE_models import VAE
from sklearn.utils import shuffle
import random
from utils import load_data, accuracy, new_load_data
import os
import pickle
import numpy as np
from evaluation import RECALL_NDCG


# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
method_name = 'VAE'
train_fts_ratio = 0.4*1.0
topK_list = [3, 5, 10]

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='steam', help='cora, citeseer, steam')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# set loss instances from classes
BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE = torch.nn.MSELoss(reduction='none')

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_discrete(recon_x, x, mu, logvar, pos_weight_tensor, neg_weight_tensor, is_test=False):
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, pos_weight_tensor, neg_weight_tensor)

    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)

    if not is_test:
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_bce + loss_kld
    else:
        return loss_bce

def loss_function_continuous(recon_x, x, mu, logvar, pos_weight_tensor, neg_weight_tensor):
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])

    loss_mse = torch.mean(MSE(output_fts_reshape, out_fts_lbls_reshape))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss_mse + loss_kld

def save_generative_fts(aggregation_z):
    model.eval()
    output_fts = model.decode(aggregation_z)

    # find non featured nodes positions
    non_featured_idx = (aggregation_z.sum(1) == 0).nonzero().reshape([-1])

    if args.dataset in ['cora', 'citeseer', 'steam']:
        output_fts = torch.sigmoid(output_fts)
    elif args.dataset in ['pubmed']:
        output_fts = output_fts

    output_fts[non_featured_idx] = 0.0

    if args.cuda:
        train_fts = true_features[train_fts_idx].data.cpu().numpy()
        vali_fts = true_features[vali_fts_idx].data.cpu().numpy()
        test_fts = output_fts.data.cpu().numpy()

        train_fts_idx_arr = train_fts_idx.cpu().numpy()
        vali_fts_idx_arr = vali_fts_idx.cpu().numpy()
        test_fts_idx_arr = test_fts_idx.cpu().numpy()

    else:
        train_fts = true_features[train_fts_idx].data.numpy()
        vali_fts = true_features[vali_fts_idx].data.numpy()
        test_fts = output_fts.data.numpy()

        train_fts_idx_arr = train_fts_idx.numpy()
        vali_fts_idx_arr = vali_fts_idx.numpy()
        test_fts_idx_arr = test_fts_idx.numpy()


    save_fts = np.zeros(shape=true_features.shape)
    save_fts[train_fts_idx_arr] = train_fts
    save_fts[vali_fts_idx_arr] = vali_fts
    save_fts[test_fts_idx_arr] = test_fts

    print('Saving generated features and true features......')
    pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))

if __name__ == "__main__":
    # Load data
    print('loading dataset: {}'.format(args.dataset))
    # note that the node_class_lbls, node_idx_train, node_idx_val, node_idx_test are only used for evaluation.
    adj, true_features, node_class_lbls, _, _, _ = new_load_data(args.dataset, norm_adj=False, generative_flag=True)

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

    # set loss function and pos weight
    if args.dataset in ['cora', 'citeseer', 'steam']:
        loss_function = loss_function_discrete
        pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item() / (torch.sum(true_features[train_fts_idx] != 0.0).item())
    elif args.dataset in ['pubmed']:
        loss_function = loss_function_continuous
        pos_weight = 1.0

    if args.cuda:
        pos_weight_tensor = torch.FloatTensor([pos_weight]).cuda()
        neg_weight_tensor = torch.FloatTensor([1.0]).cuda()
    else:
        pos_weight_tensor = torch.FloatTensor([pos_weight])
        neg_weight_tensor = torch.FloatTensor([1.0])

    loss_values = []
    best = 10000.0
    bad_counter = 0
    model = VAE(n_fts=true_features.shape[1], n_hid=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    eva_values_list = []
    best_recall = 0.0
    best_mse = 100000.0
    for epoch in range(1, args.epochs + 1):
        # train model
        model.train()
        optimizer.zero_grad()

        train_fts = true_features[train_fts_idx]
        train_fts = train_fts.to(device)
        vali_fts = true_features[vali_fts_idx]
        vali_fts = vali_fts.to(device)

        recon_batch, mu, logvar = model(train_fts)
        loss = loss_function(recon_batch, train_fts, mu, logvar, pos_weight_tensor, neg_weight_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        recon_batch, mu, logvar = model(vali_fts)
        loss_val = loss_function(recon_batch, vali_fts, mu, logvar, pos_weight_tensor, neg_weight_tensor)

        if epoch % 1 == 0:
            print('Train Epoch: {}, train loss: {:.8f}, vali loss: {:.8f}'.format(epoch, loss.item(), loss_val.item()))

        loss_val_value = loss_val.item()
        loss_values.append(loss_val_value)

        if args.dataset in ['cora', 'citeseer', 'steam']:
            # find neighbors and make raw feature aggregation for unknown nodes
            temp_adj = adj.to_dense()
            temp_adj[temp_adj != 0.0] = 1.0

            mask_adj = temp_adj[vali_fts_idx, :]
            model.eval()

            model(train_fts)

            if args.cuda:
                encoded_fts = model.z.data.cpu().numpy()
                train_fts_idx_arr = train_fts_idx.cpu().numpy()
                latent_z = model.z.data.cpu()
            else:
                encoded_fts = model.z.data.numpy()
                train_fts_idx_arr = train_fts_idx.numpy()
                latent_z = model.z.data

            mask_fts = np.zeros(shape=(true_features.shape[0], encoded_fts.shape[1]))
            mask_fts[train_fts_idx_arr] = encoded_fts

            aggregation_z = torch.mm(mask_adj, torch.FloatTensor(mask_fts)) / torch.reshape(mask_adj.sum(1) + 1e-24,
                                                                                            shape=[-1, 1])
            if args.cuda:
                aggregation_z = torch.FloatTensor(aggregation_z).cuda()
            else:
                aggregation_z = torch.FloatTensor(aggregation_z)

            aggregation_fts = model.decode(aggregation_z)

            # find non featured nodes positions
            non_featured_idx = (aggregation_z.sum(1) == 0).nonzero().reshape([-1])
            aggregation_fts[non_featured_idx] = 0.0

            if args.cuda:
                aggregation_fts = aggregation_fts.data.cpu().numpy()
                gt_fts = true_features[vali_fts_idx].cpu().numpy()
            else:
                aggregation_fts = aggregation_fts.data.numpy()
                gt_fts = true_features[vali_fts_idx].numpy()

            avg_recall, avg_ndcg = RECALL_NDCG(aggregation_fts, gt_fts, topN=topK_list[0])
            eva_values_list.append(avg_recall)

            if eva_values_list[-1] > best_recall:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                            'best_gene_{}_{}.pkl'.format(args.dataset,
                                                                                         train_fts_ratio)))
                best_recall = eva_values_list[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

        elif args.dataset in ['pubmed']:
            eva_values_list.append(loss_val_value)
            if eva_values_list[-1] < best_mse:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                            'best_gene_{}_{}.pkl'.format(args.dataset, train_fts_ratio)))
                best_mse = eva_values_list[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            # if bad_counter == args.patience:
            #     break

print("VAE Optimization Finished!")
print("Train fts ratio: {}".format(train_fts_ratio))


# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'output', method_name,
                                                        'best_gene_{}_{}.pkl'.format(args.dataset, train_fts_ratio))))

'''
evaluation for Recall and NDCG
'''
# find neighbors and make raw feature aggregation for unknown nodes
temp_adj = adj.to_dense()
temp_adj[temp_adj!=0.0] = 1.0

mask_adj = temp_adj[test_fts_idx, :]

model.eval()

model(train_fts)

if args.cuda:
    encoded_fts = model.z.data.cpu().numpy()
    train_fts_idx_arr = train_fts_idx.cpu().numpy()
    latent_z = model.z.data.cpu()
else:
    encoded_fts = model.z.data.numpy()
    train_fts_idx_arr = train_fts_idx.numpy()
    latent_z = model.z.data

mask_fts = np.zeros(shape=(true_features.shape[0], encoded_fts.shape[1]))
mask_fts[train_fts_idx_arr] = encoded_fts

aggregation_z = torch.mm(mask_adj, torch.FloatTensor(mask_fts))/torch.reshape(mask_adj.sum(1)+1e-24, shape=[-1, 1])

if args.cuda:
    aggregation_z = torch.FloatTensor(aggregation_z).cuda()
else:
    aggregation_z = torch.FloatTensor(aggregation_z)

# pickle.dump(aggregation_z, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_latent_Z.pkl'.format(
#     args.dataset, train_fts_ratio)), 'wb'))


aggregation_fts = model.decode(aggregation_z)
# find non featured nodes positions
non_featured_idx = (aggregation_z.sum(1) == 0).nonzero().reshape([-1])
aggregation_fts[non_featured_idx] = 0.0

if args.dataset in ['cora', 'citeseer', 'steam']:
    loss_test = loss_function(aggregation_fts, true_features[test_fts_idx].to(device), mu, logvar, pos_weight_tensor, neg_weight_tensor, is_test=True)
    print('BCE loss: {}'.format(loss_test.item()))

print('test for label propagation......')
if args.cuda:
    aggregation_fts = aggregation_fts.data.cpu().numpy()
    gt_fts = true_features[test_fts_idx].cpu().numpy()
else:
    aggregation_fts = aggregation_fts.data.numpy()
    gt_fts = true_features[test_fts_idx].numpy()

if args.dataset in ['cora', 'citeseer', 'steam']:
    '''
    evaluation for Recall and NDCG
    '''
    for topK in topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(aggregation_fts, gt_fts, topN=topK)
        print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
    print('method: {}, dataset: {}'.format(method_name, args.dataset))
elif args.dataset in ['pubmed']:
    NL2 = np.mean(np.linalg.norm(aggregation_fts - gt_fts, axis=1) / np.linalg.norm(gt_fts, axis=1))
    print('normalized L2 distance: {:.8f}'.format(NL2))

'''
save necessary fts for evaluation for continuous fts
'''
# # the following needs to be revised
# save_generative_fts(aggregation_z)

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
print('method: {}, dataset: {}, ratio: {}'.format(method_name, args.dataset, train_fts_ratio))
