from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from NANG_models import LFI
from sklearn.utils import shuffle
import random
from utils import load_data, accuracy, new_load_data, MMD
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from evaluation import RECALL_NDCG

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
method_name = 'LFI'
train_fts_ratio = 0.4*1.0
topK_list = [10, 20, 50]

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='cora', help='cora, citeseer, steam')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--enc-name', type=str, default='GCN', help='Initial encoder model, GCN or GAT')
parser.add_argument('--alpha', type=float, default=0.2, help='Initial alpha for leak relu when use GAT as enc-name')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--neg_times', type=int, default=1, help='neg times with the positive pairs')
parser.add_argument('--n_gene', type=int, default=2, help='epoch number of generator')
parser.add_argument('--n_disc', type=int, default=1, help='epoch number of dsiscriminator')
parser.add_argument('--lambda_recon', type=float, default=1.0, help='lambda for reconstruction, always 1.0 in our model')
parser.add_argument('--lambda_cross', type=float, default=10.0, help='lambda for cross stream')
parser.add_argument('--lambda_gan', type=float, default=1.0, help='lambda for GAN loss, always 1.0 in our model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
print('beging...............')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# set loss instances from classes
BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE = torch.nn.MSELoss(reduction='none')


# Reconstruction + KL divergence losses summed over all elements and batch
def graph_loss_func(graph_recon=None, pos_indices=None, neg_indices=None,
                    pos_values=None, neg_values=None):
    loss_indices = torch.cat([pos_indices, neg_indices], dim=0)
    preds_logits = graph_recon[loss_indices[:, 0], loss_indices[:, 1]]
    labels = torch.cat([pos_values, neg_values])
    loss_bce = torch.mean(BCE(preds_logits, labels))
    return loss_bce



# Reconstruction + KL divergence losses summed over all elements and batch
def fts_loss_discrete(recon_x=None, x=None, pos_weight_tensor=None, neg_weight_tensor=None):
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, pos_weight_tensor, neg_weight_tensor)

    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)

    return loss_bce


def fts_loss_continuous(recon_x=None, x=None, pos_weight_tensor=None, neg_weight_tensor=None):
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])

    loss_mse = torch.mean(MSE(output_fts_reshape, out_fts_lbls_reshape))

    return loss_mse


def save_generative_fts(gene_fts):
    if args.dataset in ['cora', 'citeseer', 'steam']:
        output_fts = 1.0/(1.0+np.exp(-gene_fts))
    elif args.dataset in ['pubmed']:
        output_fts = gene_fts

    if args.cuda:
        train_fts = true_features[train_fts_idx].data.cpu().numpy()
        vali_fts = true_features[vali_fts_idx].data.cpu().numpy()

        train_fts_idx_arr = train_fts_idx.cpu().numpy()
        vali_fts_idx_arr = vali_fts_idx.cpu().numpy()
        test_fts_idx_arr = test_fts_idx.cpu().numpy()

    else:
        train_fts = true_features[train_fts_idx].data.numpy()
        vali_fts = true_features[vali_fts_idx].data.numpy()

        train_fts_idx_arr = train_fts_idx.numpy()
        vali_fts_idx_arr = vali_fts_idx.numpy()
        test_fts_idx_arr = test_fts_idx.numpy()

    save_fts = np.zeros(shape=true_features.shape)
    save_fts[train_fts_idx_arr] = train_fts
    save_fts[vali_fts_idx_arr] = vali_fts
    save_fts[test_fts_idx_arr] = output_fts

    print('Saving generated features and true features......')
    pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', method_name,
                                            'gene_fts_train_ratio_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset, train_fts_ratio,
                                                                                            args.lambda_gan, args.lambda_recon,
                                                                                                args.lambda_cross)), 'wb'))


if __name__ == "__main__":
    # Load data
    print('loading dataset: {}'.format(args.dataset))
    # note that the node_class_lbls, node_idx_train, node_idx_val, node_idx_test are only used for evaluation.
    adj, true_features, node_class_lbls, _, _, _ = new_load_data(args.dataset, norm_adj=False, generative_flag=True)

    pickle.dump(adj.to_dense().numpy(), open(os.path.join(os.getcwd(), 'features', method_name,
                                                '{}_sp_adj.pkl'.format(args.dataset)), 'wb'))
    pickle.dump(node_class_lbls.numpy(), open(os.path.join(os.getcwd(), 'data', args.dataset, '{}_labels.pkl'.format(args.dataset)), 'wb'))

    norm_adj, _, _, _, _, _ = new_load_data(args.dataset, norm_adj=True, generative_flag=True)
    norm_adj_arr = norm_adj.to_dense().numpy()

    # generate ont-hot features for all nodes, this means no node feature is used
    indices = torch.LongTensor(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0))
    values = torch.FloatTensor(np.ones(indices.shape[1]))
    diag_fts = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))

    # split train features and generative features
    shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=args.seed)
    train_fts_idx = torch.LongTensor(shuffled_nodes[:int(train_fts_ratio * adj.shape[0])])
    vali_fts_idx = torch.LongTensor(
        shuffled_nodes[int(0.4 * adj.shape[0]):int((0.4 + 0.1) * adj.shape[0])])
    test_fts_idx = torch.LongTensor(shuffled_nodes[int((0.4 + 0.1) * adj.shape[0]):])

    # make files
    if not os.path.exists(os.path.join(os.getcwd(), 'features', method_name)):
        os.makedirs(os.path.join(os.getcwd(), 'features', method_name))

    if not os.path.exists(os.path.join(os.getcwd(), 'output', method_name)):
        os.makedirs(os.path.join(os.getcwd(), 'output', method_name))

    pickle.dump(train_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_train_fts_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))
    pickle.dump(vali_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_vali_fts_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))
    pickle.dump(test_fts_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_test_fts_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))

    # set loss function and pos weight
    if args.dataset in ['cora', 'citeseer', 'steam']:
        fts_loss_func = fts_loss_discrete
        pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item() / (
            torch.sum(true_features[train_fts_idx] != 0.0).item())
    elif args.dataset in ['pubmed']:
        fts_loss_func = fts_loss_continuous
        pos_weight = 1.0

    if args.cuda:
        pos_weight_tensor = torch.FloatTensor([pos_weight]).cuda()
        neg_weight_tensor = torch.FloatTensor([1.0]).cuda()
    else:
        pos_weight_tensor = torch.FloatTensor([pos_weight])
        neg_weight_tensor = torch.FloatTensor([1.0])

    '''
    make data preparation
    '''
    if args.cuda:
        norm_adj = norm_adj.cuda()
        diag_fts = diag_fts.cuda()
        train_fts = true_features[train_fts_idx].cuda()
        vali_fts = true_features[vali_fts_idx].cuda()
        test_fts = true_features[test_fts_idx].cuda()
    else:
        train_fts = true_features[train_fts_idx]
        vali_fts = true_features[vali_fts_idx]
        test_fts = true_features[test_fts_idx]

    #  set params for graph loss
    n_pos = len(norm_adj._values())

    if args.cuda:
        pos_indices = norm_adj._indices().cpu().numpy()
    else:
        pos_indices = norm_adj._indices().numpy()
    pos_indices = list(zip(pos_indices[0, :], pos_indices[1, :]))

    if not os.path.exists(os.path.join(os.getcwd(), 'data', args.dataset, '{}_{}_neg_indices.pkl'.format(args.dataset, train_fts_ratio))):
        zero_indices = np.where(norm_adj_arr == 0)
        neg_indices = list(zip(zero_indices[0], zero_indices[1]))
        neg_indices = shuffle(neg_indices, random_state=args.seed)[:args.neg_times * n_pos]
        pickle.dump(neg_indices, open(os.path.join(os.getcwd(), 'data', args.dataset,
                                                   '{}_{}_neg_indices.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))
    else:
        neg_indices = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset,
                                                    '{}_{}_neg_indices.pkl'.format(args.dataset, train_fts_ratio)), 'rb'))

    if args.cuda:
        neg_indices = torch.LongTensor(neg_indices).cuda()
        neg_values = torch.zeros(size=[len(neg_indices)]).cuda()
        pos_values = torch.ones(size=[len(pos_indices)]).cuda()
        pos_indices = torch.LongTensor(pos_indices).cuda()
    else:
        neg_indices = torch.LongTensor(neg_indices)
        neg_values = torch.zeros(size=[len(neg_indices)])
        pos_values = torch.ones(size=[len(pos_indices)])
        pos_indices = torch.LongTensor(pos_indices)

    '''
    # define train adj subset of cross loss for A
    '''
    if args.cuda:
        train_fts_idx_arr = train_fts_idx.cpu().numpy()
    else:
        train_fts_idx_arr = train_fts_idx.numpy()

    # make sub indices for training process
    sub_norm_adj = norm_adj_arr[train_fts_idx_arr, :]
    sub_norm_adj = sub_norm_adj[:, train_fts_idx_arr]
    sub_pos_indices = np.where(sub_norm_adj != 0.0)
    sub_pos_indices = list(zip(sub_pos_indices[0], sub_pos_indices[1]))

    if not os.path.exists(os.path.join(os.getcwd(), 'data', args.dataset, '{}_{}_sub_neg_indices.pkl'.format(args.dataset, train_fts_ratio))):
        sub_zero_indices = np.where(sub_norm_adj==0)
        sub_neg_indices = list(zip(sub_zero_indices[0], sub_zero_indices[1]))
        sub_neg_indices = shuffle(sub_neg_indices, random_state=args.seed)[:args.neg_times * len(sub_pos_indices)]
        pickle.dump(sub_neg_indices, open(os.path.join(os.getcwd(), 'data', args.dataset,
                                                       '{}_{}_sub_neg_indices.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))
    else:
        sub_neg_indices = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset,
                                                       '{}_{}_sub_neg_indices.pkl'.format(args.dataset,
                                                                                          train_fts_ratio)), 'rb'))


    if args.cuda:
        sub_neg_indices = torch.LongTensor(sub_neg_indices).cuda()
        sub_neg_values = torch.zeros(size=[len(sub_neg_indices)]).cuda()
        sub_pos_values = torch.ones(size=[len(sub_pos_indices)]).cuda()
        sub_pos_indices = torch.LongTensor(sub_pos_indices).cuda()
    else:
        sub_neg_indices = torch.LongTensor(sub_neg_indices)
        sub_neg_values = torch.zeros(size=[len(sub_neg_indices)])
        sub_pos_values = torch.ones(size=[len(sub_pos_indices)])
        sub_pos_indices = torch.LongTensor(sub_pos_indices)

    '''
    # define vali adj subset of cross loss for A
    '''
    if args.cuda:
        vali_fts_idx_arr = vali_fts_idx.cpu().numpy()
    else:
        vali_fts_idx_arr = vali_fts_idx.numpy()


    # make sub indices for vali process
    vali_sub_norm_adj = norm_adj_arr[vali_fts_idx_arr, :]
    vali_sub_norm_adj = vali_sub_norm_adj[:, vali_fts_idx_arr]
    vali_sub_pos_indices = np.where(vali_sub_norm_adj != 0.0)
    vali_sub_pos_indices = list(zip(vali_sub_pos_indices[0], vali_sub_pos_indices[1]))

    if not os.path.exists(os.path.join(os.getcwd(), 'data', args.dataset,
                                       '{}_{}_vali_sub_neg_indices.pkl'.format(args.dataset, train_fts_ratio))):
        vali_sub_all_indices = []
        for i in range(vali_sub_norm_adj.shape[0]):
            for j in range(i, vali_sub_norm_adj.shape[0]):
                vali_sub_all_indices.append((i, j))
                vali_sub_all_indices.append((j, i))
        vali_sub_neg_indices = list(set(vali_sub_all_indices) - set(vali_sub_pos_indices))
        vali_sub_neg_indices = shuffle(vali_sub_neg_indices, random_state=args.seed)[:args.neg_times * len(vali_sub_pos_indices)]
        pickle.dump(vali_sub_neg_indices, open(os.path.join(os.getcwd(), 'data', args.dataset,
                                       '{}_{}_vali_sub_neg_indices.pkl'.format(args.dataset, train_fts_ratio)), 'wb'))
    else:
        vali_sub_neg_indices = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset,
                                                            '{}_{}_vali_sub_neg_indices.pkl'.format(args.dataset,
                                                            train_fts_ratio)), 'rb'))



    if args.cuda:
        vali_sub_neg_indices = torch.LongTensor(vali_sub_neg_indices).cuda()
        vali_sub_neg_values = torch.zeros(size=[len(vali_sub_neg_indices)]).cuda()
        vali_sub_pos_values = torch.ones(size=[len(vali_sub_pos_indices)]).cuda()
        vali_sub_pos_indices = torch.LongTensor(vali_sub_pos_indices).cuda()
    else:
        vali_sub_neg_indices = torch.LongTensor(vali_sub_neg_indices)
        vali_sub_neg_values = torch.zeros(size=[len(vali_sub_neg_indices)])
        vali_sub_pos_values = torch.ones(size=[len(vali_sub_pos_indices)])
        vali_sub_pos_indices = torch.LongTensor(vali_sub_pos_indices)

    '''
    define things for LFI model
    '''
    prior = torch.distributions.normal.Normal(loc=torch.FloatTensor([0.0]), scale=torch.FloatTensor([1.0]))
    model = LFI(n_nodes=norm_adj.shape[0], n_fts=true_features.shape[1], n_hid=args.hidden, dropout=args.dropout, args=args)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
                             weight_decay=args.weight_decay)


    train_G_loss_list = []
    train_D_loss_list = []
    vali_G_loss_list = []
    vali_D_loss_list = []
    joint_loss_list = []
    eva_values_list = []
    train_MMD_list = []
    vali_MMD_list = []

    # set params to calculate MMD distance
    sigma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4]
    sigma_list = torch.FloatTensor(np.array(sigma_list))
    if args.cuda:
        sigma_list = sigma_list.cuda()

    best = 0.0
    best_mse = 10000.0
    bad_counter = 0
    best_epoch= 0

    if norm_adj.is_sparse and args.enc_name=='GAT':
        norm_adj = norm_adj.to_dense()
    for epoch in range(1, args.epochs + 1):
        '''
        train the generators
        '''

        for ele in model.named_parameters():
            name = ele[0]
            if 'disc' in name:
                ele[1].requires_grad = False
            else:
                ele[1].requires_grad = True

        for i in range(1, args.n_gene + 1):
            # train model
            model.train()
            optimizer.zero_grad()

            ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj = model(train_fts, norm_adj, diag_fts)


            fts_ae_loss = args.lambda_recon*fts_loss_func(ae_fts, train_fts, pos_weight_tensor, neg_weight_tensor)
            fts_gae_loss = args.lambda_cross*fts_loss_func(gae_fts[train_fts_idx], train_fts, pos_weight_tensor, neg_weight_tensor)
            adj_ae_loss = args.lambda_cross*graph_loss_func(graph_recon=ae_adj, pos_indices=sub_pos_indices, neg_indices=sub_neg_indices,
                                           pos_values=sub_pos_values, neg_values=sub_neg_values)
            adj_gae_loss = args.lambda_recon*graph_loss_func(graph_recon=gae_adj, pos_indices=pos_indices, neg_indices=neg_indices,
                                           pos_values=pos_values, neg_values=neg_values)

            fake_logits_ae = model.disc(ae_z).reshape([-1])
            fake_logits_gae = model.disc(gae_z[train_fts_idx]).reshape([-1])

            G_lbls_1 = torch.ones_like(fake_logits_ae)

            G_loss_ae = BCE(fake_logits_ae, G_lbls_1).mean()
            G_loss_gae = BCE(fake_logits_gae, G_lbls_1).mean()

            G_loss = args.lambda_gan*(G_loss_ae + G_loss_gae)

            joint_loss = fts_ae_loss + fts_gae_loss + adj_ae_loss + adj_gae_loss

            gene_loss = fts_ae_loss + fts_gae_loss + adj_ae_loss + adj_gae_loss + G_loss

            gene_loss.backward()
            optimizer.step()

        '''
        train the discriminator
        '''
        for ele in model.named_parameters():
            name = ele[0]
            if 'disc' in name:
                ele[1].requires_grad = True
            else:
                ele[1].requires_grad = False

        for i in range(1, args.n_disc + 1):
            # train model
            model.train()
            optimizer.zero_grad()

            ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj = model(train_fts, norm_adj, diag_fts)
            # Sample noise as discriminator ground truth
            # standard Gaussian
            true_z = prior.sample([ae_z.shape[0], ae_z.shape[1]]).reshape([ae_z.shape[0], ae_z.shape[1]])

            if args.cuda:
                true_z = true_z.cuda()
            true_logits_ae = model.disc(true_z).reshape([-1])
            true_logits_gae = model.disc(true_z).reshape([-1])
            fake_logits_ae = model.disc(ae_z).reshape([-1])
            fake_logits_gae = model.disc(gae_z[train_fts_idx]).reshape([-1])

            logits_ae = torch.cat([true_logits_ae, fake_logits_ae])
            logits_gae = torch.cat([true_logits_gae, fake_logits_gae])

            D_lbls_10 = torch.cat([torch.ones_like(true_logits_ae), torch.zeros_like(fake_logits_ae)])

            D_loss_ae = BCE(logits_ae, D_lbls_10).mean()
            D_loss_gae = BCE(logits_gae, D_lbls_10).mean()

            D_loss = args.lambda_gan*(D_loss_ae + D_loss_gae)

            D_loss.backward()
            optimizer.step()

        train_D_loss_list.append(D_loss.item()/args.lambda_gan)
        train_G_loss_list.append(G_loss.item()/args.lambda_gan)

        joint_loss_list.append(joint_loss.item())


        # make evaluation process
        model.eval()

        # get MMD distance for two distributions
        train_ae_z, _, _, train_gae_z, _, _ = model(train_fts, norm_adj, diag_fts)

        train_mmd = 0.5*(MMD(train_ae_z, true_z, sigma_list) + MMD(train_gae_z[train_fts_idx], true_z, sigma_list))
        # if train_mmd > 100.0:
        #     train_MMD_list.append(train_MMD_list[-1])
        # else:
        train_MMD_list.append(train_mmd.item())

        ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj = model(vali_fts, norm_adj, diag_fts)
        temp_z = prior.sample([ae_z.shape[0], ae_z.shape[1]]).reshape([ae_z.shape[0], ae_z.shape[1]])
        if args.cuda:
            temp_z = temp_z.cuda()
        vali_mmd = 0.5 * (MMD(ae_z, temp_z, sigma_list) + MMD(gae_z[vali_fts_idx], temp_z, sigma_list))
        # if vali_mmd > 100.0:
        #     vali_MMD_list.append(vali_MMD_list[-1])
        # else:
        vali_MMD_list.append(vali_mmd.item())

        # get generator loss
        vali_fts_ae_loss = args.lambda_recon*fts_loss_func(ae_fts, vali_fts, pos_weight_tensor, neg_weight_tensor)
        vali_fts_gae_loss = args.lambda_cross*fts_loss_func(gae_fts[vali_fts_idx], vali_fts, pos_weight_tensor, neg_weight_tensor)
        vali_adj_ae_loss = args.lambda_cross*graph_loss_func(graph_recon=ae_adj, pos_indices=vali_sub_pos_indices, neg_indices=vali_sub_neg_indices,
                                      pos_values=vali_sub_pos_values, neg_values=vali_sub_neg_values)
        vali_adj_gae_loss = args.lambda_recon*graph_loss_func(graph_recon=gae_adj, pos_indices=pos_indices, neg_indices=neg_indices,
                                       pos_values=pos_values, neg_values=neg_values)

        fake_logits_ae = model.disc(ae_z).reshape([-1])
        fake_logits_gae = model.disc(gae_z[vali_fts_idx]).reshape([-1])

        G_lbls_1 = torch.ones_like(fake_logits_ae)
        vali_G_loss_ae = BCE(fake_logits_ae, G_lbls_1).mean()
        vali_G_loss_gae = BCE(fake_logits_gae, G_lbls_1).mean()

        vali_G_loss = args.lambda_gan*(vali_G_loss_ae + vali_G_loss_gae)

        vali_gene_loss = vali_fts_ae_loss + vali_fts_gae_loss + vali_adj_ae_loss + vali_adj_gae_loss + vali_G_loss

        # discriminator loss
        # Sample noise as discriminator ground truth
        # standard Gaussian
        true_z = prior.sample([ae_z.shape[0], ae_z.shape[1]]).reshape([ae_z.shape[0], ae_z.shape[1]])
        if args.cuda:
            true_z = true_z.cuda()

        true_logits_ae = model.disc(true_z).reshape([-1])
        true_logits_gae = model.disc(true_z).reshape([-1])
        fake_logits_ae = model.disc(ae_z).reshape([-1])
        fake_logits_gae = model.disc(gae_z[vali_fts_idx]).reshape([-1])

        logits_ae = torch.cat([true_logits_ae, fake_logits_ae])
        logits_gae = torch.cat([true_logits_gae, fake_logits_gae])

        D_lbls_10 = torch.cat([torch.ones_like(true_logits_ae), torch.zeros_like(fake_logits_ae)])

        vali_D_loss_ae = BCE(logits_ae, D_lbls_10).mean()
        vali_D_loss_gae = BCE(logits_gae, D_lbls_10).mean()

        vali_D_loss = args.lambda_gan*(vali_D_loss_ae + vali_D_loss_gae)

        vali_G_loss_list.append(vali_G_loss.item()/args.lambda_gan)
        vali_D_loss_list.append(vali_D_loss.item()/args.lambda_gan)

        '''
        make early stop condition
        lbls
        '''

        if args.dataset in ['cora', 'citeseer', 'steam']:
            # make validation for evaluation metric
            gene_fts_sigmoid = torch.sigmoid(gae_fts[vali_fts_idx])
            if args.cuda:
                gene_fts_sigmoid = gene_fts_sigmoid.data.cpu().numpy()
                gt_fts = true_features[vali_fts_idx].cpu().numpy()
            else:
                gene_fts_sigmoid = gene_fts_sigmoid.data.numpy()
                gt_fts = true_features[vali_fts_idx].numpy()

            avg_recall, avg_ndcg = RECALL_NDCG(gene_fts_sigmoid, gt_fts, topN=topK_list[0])
            eva_values_list.append(avg_recall)
            if eva_values_list[-1] > best:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                                'best_LFI_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset,
                                                                                                    train_fts_ratio,
                                                                                                    args.lambda_gan,
                                                                                                    args.lambda_recon,
                                                                                                    args.lambda_cross)))
                best = eva_values_list[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            # if bad_counter == args.patience:
            #     break

        elif args.dataset in ['pubmed']:
            if args.lambda_cross == 0.0:
                eva_values_list.append(vali_fts_ae_loss.item()/args.lambda_recon)
            else:
                eva_values_list.append(vali_fts_gae_loss.item() / args.lambda_cross)

            if eva_values_list[-1] < best_mse:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                                'best_LFI_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset,
                                                                                                    train_fts_ratio,
                                                                                                    args.lambda_gan,
                                                                                                    args.lambda_recon,
                                                                                                    args.lambda_cross)))
                best_mse = eva_values_list[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

        '''
        print training and validation information
        '''
        if epoch % 1 == 0:
            print('Train Epoch: {}, gene loss: {:.8f}, fts ae loss: {:.8f}, fts gae loss: {:.8f}, adj ae loss: {:.8f}, '
                  'adj gae loss: {:.8f}, G loss: {:.8f}, D loss: {:.8f}'.format(
                    epoch, gene_loss.item(), fts_ae_loss.item(), fts_gae_loss.item(), adj_ae_loss.item(), adj_gae_loss.item(),
                    G_loss.item(), D_loss.item()))

            print('Vali Epoch: {}, gene loss: {:.8f}, fts ae loss: {:.8f}, fts gae loss: {:.8f}, adj ae loss: {:.8f}, '
                  'adj gae loss: {:.8f}, G loss: {:.8f}, D loss: {:.8f}, eva_values: {:.8f}'.format(
                    epoch, vali_gene_loss.item(), vali_fts_ae_loss.item(), vali_fts_gae_loss.item(), vali_adj_ae_loss.item(),
                vali_adj_gae_loss.item(), vali_G_loss.item(), vali_D_loss.item(), eva_values_list[-1]))


print("LFI Optimization Finished!")
print("Train fts ratio: {}, best epoch: {}".format(train_fts_ratio, best_epoch))

pickle.dump(joint_loss_list, open(os.path.join(os.getcwd(), 'features', method_name, '{}_train_joint_loss_list_G{}_C{}_R{}.pkl'.format(args.dataset,
                                                                                                                               args.lambda_gan,
                                                                                                                            args.lambda_cross,
                                                                                                                            args.lambda_recon)), 'wb'))


pickle.dump(train_MMD_list, open(os.path.join(os.getcwd(), 'features', method_name, '{}_train_MMD_list_G{}_C{}_R{}.pkl'.format(args.dataset,
                                                                                                                               args.lambda_gan,
                                                                                                                            args.lambda_cross,
                                                                                                                            args.lambda_recon)), 'wb'))

pickle.dump(vali_MMD_list, open(os.path.join(os.getcwd(), 'features', method_name, '{}_vali_MMD_list_G{}_C{}_R{}.pkl'.format(args.dataset,
                                                                                                                             args.lambda_gan,
                                                                                                                            args.lambda_cross,
                                                                                                                            args.lambda_recon)), 'wb'))

# # plot loss curve
# font = {'family': 'Times New Roman',
#         'color': 'black',
#         'weight': 'bold',
#         'size': 15,
#         }
# mycolor = np.array([[224, 32, 32],
#                     [255, 192, 0],
#                     [32, 160, 64],
#                     [48, 96, 192],
#                     [192, 48, 192]]) / 255.0
# mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']
#
# my_line_width = 3
# my_marker_size = 10
# #
# # plot train G/D curve
# plt.figure(1)
# plt.style.use('ggplot')
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
# x_axix = range(len(train_G_loss_list))
# plt.plot(x_axix, train_G_loss_list, color='orange', label='Generator', linewidth=my_line_width,
#          markersize=my_marker_size)
# plt.plot(x_axix, train_D_loss_list, color='blue', label='Discriminator', linewidth=my_line_width,
#          markersize=my_marker_size)
# my_legend = plt.legend(loc='upper right', fontsize=15)
# frame = my_legend.get_frame()
# frame.set_alpha(1)
# frame.set_facecolor('none')
#
# plt.tick_params(labelsize='10')
# plt.xlabel('Epoch', fontdict=font)
# plt.ylabel('Train GAN Loss', fontdict=font)
# # plt.show()
# plt.savefig(
#     os.path.join(os.getcwd(), 'figures', method_name, '{}_{}_G{}_R{}_C{}_train_GAN_loss.png'.format(args.dataset,
#                                                                                                 train_fts_ratio,
#                                                                                                 args.lambda_gan,
#                                                                                                 args.lambda_recon,
#                                                                                                 args.lambda_cross)))
#
# # plot vali G/D curve
# plt.figure(2)
# plt.style.use('ggplot')
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
# x_axix = range(len(vali_G_loss_list))
# plt.plot(x_axix, vali_G_loss_list, color='orange', label='Generator', linewidth=my_line_width,
#          markersize=my_marker_size)
# plt.plot(x_axix, vali_D_loss_list, color='blue', label='Discriminator', linewidth=my_line_width,
#          markersize=my_marker_size)
# my_legend = plt.legend(loc='upper right', fontsize=15)
# frame = my_legend.get_frame()
# frame.set_alpha(1)
# frame.set_facecolor('none')
#
# plt.tick_params(labelsize='10')
# plt.xlabel('Epoch', fontdict=font)
# plt.ylabel('Vali GAN Loss', fontdict=font)
# # plt.show()
# plt.savefig(
#     os.path.join(os.getcwd(), 'figures', method_name, '{}_{}_G{}_R{}_C{}_vali_GAN_loss.png'.format(args.dataset,
#                                                                                                train_fts_ratio,
#                                                                                                args.lambda_gan,
#                                                                                                args.lambda_recon,
#                                                                                                args.lambda_cross)))
#
#
# # plot train loss curve
# plt.figure(3)
# plt.style.use('ggplot')
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
# x_axix = range(len(joint_loss_list))
# plt.plot(x_axix, joint_loss_list, color=mycolor[0], linewidth=my_line_width,
#          markersize=my_marker_size)
# plt.tick_params(labelsize='10')
# plt.xlabel('Epoch', fontdict=font)
# plt.ylabel('Train Joint Loss', fontdict=font)
# # plt.show()
# plt.savefig(
#     os.path.join(os.getcwd(), 'figures', method_name, '{}_{}_G{}_R{}_C{}_train_joint_loss.png'.format(args.dataset,
#                                                                                                   train_fts_ratio,
#                                                                                                   args.lambda_gan,
#                                                                                                     args.lambda_recon,
#                                                                                                   args.lambda_cross)))
#
# #
# # plot vali evaluation curve
# if args.dataset in ['cora', 'citeseer', 'steam']:
#     eva_metric = 'Recall@{}'.format(topK_list[0])
# elif args.dataset in ['pubmed']:
#     eva_metric = 'RMSE'
#
# plt.figure(4)
# plt.style.use('ggplot')
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
# x_axix = range(len(eva_values_list))
# plt.plot(x_axix, eva_values_list, color=mycolor[0], linewidth=my_line_width,
#          markersize=my_marker_size)
# plt.tick_params(labelsize='10')
# plt.xlabel('Epoch', fontdict=font)
# plt.ylabel('Val {}'.format(eva_metric), fontdict=font)
# # plt.show()
# plt.savefig(
#     os.path.join(os.getcwd(), 'figures', method_name, '{}_{}_G{}_R{}_C{}_vali_eva_values.png'.format(args.dataset,
#                                                                                                      train_fts_ratio,
#                                                                                                      args.lambda_gan,
#                                                                                                      args.lambda_recon,
#                                                                                                      args.lambda_cross)))

#
# Restore best model
print('Loading {}th epoch, D loss: {:.4f}, G loss: {:.4f}'.format(best_epoch, vali_D_loss_list[best_epoch], vali_G_loss_list[best_epoch]))
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'output', method_name,
                                              'best_LFI_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset,
                                                                                      train_fts_ratio,
                                                                                      args.lambda_gan,
                                                                                      args.lambda_recon,
                                                                                      args.lambda_cross))))

'''
evaluation for Recall and NDCG
'''
# find neighbors and make raw feature aggregation for unknown nodes
model.eval()
ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj = model(train_fts, norm_adj, diag_fts)

if args.dataset in ['cora', 'citeseer', 'steam']:
    test_fts_gae_loss = fts_loss_func(gae_fts[test_fts_idx], test_fts, pos_weight_tensor, neg_weight_tensor)
    print('BCE loss: {}'.format(test_fts_gae_loss.item()))


if args.lambda_recon!=0 and args.lambda_cross!=0:
    pickle.dump(ae_z, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_latent_ae_Z.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))
    pickle.dump(gae_z, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_latent_gae_Z.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))

gene_fts = gae_fts[test_fts_idx]

print('test for label propagation......')
if args.cuda:
    gene_fts = gene_fts.data.cpu().numpy()
    gt_fts = true_features[test_fts_idx].cpu().numpy()
else:
    gene_fts = gene_fts.data.numpy()
    gt_fts = true_features[test_fts_idx].numpy()


if args.dataset in ['cora', 'citeseer', 'steam']:
    '''
    evaluation for Recall and NDCG
    '''
    for topK in topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=topK)
        print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
    print('method: {}, dataset: {}'.format(method_name, args.dataset))
elif args.dataset in ['pubmed']:
    NL2 = np.mean(np.linalg.norm(gene_fts - gt_fts, axis=1) / np.linalg.norm(gt_fts, axis=1))
    print('normalized L2 distance: {:.8f}'.format(NL2))

print('method: {}, dataset: {}, lambda GAN: {}, lambda cross: {}, hidden: {}'.format(method_name, args.dataset,
                                                                                     args.lambda_gan,
                                                                                     args.lambda_cross,
                                                                                     args.hidden))

'''
save necessary fts for evaluation for continuous fts
'''
# the following needs to be revised
save_generative_fts(gene_fts)

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

pickle.dump(true_features, open(os.path.join(os.getcwd(), 'features', method_name,
                                             '{}_true_features.pkl'.format(args.dataset)), 'wb'))
print('method: {}, dataset: {}, ratio: {}'.format(method_name, args.dataset, train_fts_ratio))
