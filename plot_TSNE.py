import os
import matplotlib.pyplot as plt
from evaluation import class_eva
import pickle
from sklearn.utils import shuffle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
import seaborn as sns

'''
This is the script for plot the latent embedding visualization
Only for datasets: cora, citeseer, pubmed, reddit
Only for methods: GCN, VAE, LFI
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

dataset = 'cora'  # cora, citeseer, pubmed, reddit
method_name = 'VAE'  # GCN, VAE, LFI
train_fts_ratio = 0.4

train_fts_idx = pickle.load(open(os.path.join(os.getcwd(), 'features', 'NeighAggre',
                                            '{}_{}_test_fts_idx.pkl'.format(dataset, train_fts_ratio)), 'rb'))

if method_name=='LFI':
    train_z = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_{}_latent_gae_Z.pkl'.format(dataset, train_fts_ratio)), 'rb'))
else:
    train_z = pickle.load(open(os.path.join(os.getcwd(), 'features', method_name,
                                            '{}_{}_latent_Z.pkl'.format(dataset, train_fts_ratio)), 'rb'))

all_labels = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset,
                                            '{}_labels.pkl'.format(dataset)), 'rb'))

train_fts_idx = train_fts_idx.numpy()
try:
    train_z = train_z.data.cpu().numpy()
except Exception:
    train_z = train_z.data.numpy()

if method_name == 'VAE':
    train_z = train_z
else:
    train_z = train_z[train_fts_idx]

train_fts_labels = all_labels[train_fts_idx]
domain_num = max(train_fts_labels)+1
reduced_z = TSNE(n_components=2).fit_transform(train_z)


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

palette = np.array(sns.color_palette('hls', domain_num))

my_line_width = 3
my_marker_size = 10

label_z_dict = {}
for label in list(set(train_fts_labels)):
    z_for_label = reduced_z[np.where(train_fts_labels==label)]
    label_z_dict[label] = z_for_label

plt.figure(1)
for label in label_z_dict:
    plt.scatter(label_z_dict[label][:, 0], label_z_dict[label][:, 1], s=20, color=palette[label])
# plt.show()
plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_{}_tsne.png'.format(dataset, method_name)))

