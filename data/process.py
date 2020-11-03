import os
import sys
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
from sklearn.utils import shuffle
import scipy.sparse as sp
import bson
from scipy import sparse
from scipy.linalg import svd
import gzip
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import  _pickle as cPickle
import os
import numpy as np
import scipy.sparse as sp
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize

def to_binary_bag_of_words(features):
    """Converts TF/IDF features to binary bag-of-words features."""
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def normalize_adj(A):
    """Compute D^-1/2 * A * D^-1/2."""
    # Make sure that there are no self-loops
    A = eliminate_self_loops(A)
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def renormalize_adj(A):
    """Renormalize the adjacency matrix (as in the GCN paper)."""
    A_tilde = A.tolil()
    A_tilde.setdiag(1)
    A_tilde = A_tilde.tocsr()
    A_tilde.eliminate_zeros()
    D = np.ravel(A.sum(1))
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def row_normalize(matrix):
    """Normalize the matrix so that the rows sum up to 1."""
    return normalize(matrix, norm='l1', axis=1)


def add_self_loops(A, value=1.0):
    """Set the diagonal."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.

    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def eliminate_self_loops(G):
    G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix)
    return G


def load_dataset(data_path):
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    sparse_graph : SparseGraph
        The requested dataset in sparse format.

    """
    if not data_path.endswith('.npz'):
        data_path += '.npz'
    if os.path.isfile(data_path):
        return load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError("{} doesn't exist.".format(data_path))


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_sparse_graph_to_npz(filepath, sparse_graph):
    """Save a SparseGraph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    data_dict = {
        'adj_data': sparse_graph.adj_matrix.data,
        'adj_indices': sparse_graph.adj_matrix.indices,
        'adj_indptr': sparse_graph.adj_matrix.indptr,
        'adj_shape': sparse_graph.adj_matrix.shape
    }
    if sp.isspmatrix(sparse_graph.attr_matrix):
        data_dict['attr_data'] = sparse_graph.attr_matrix.data
        data_dict['attr_indices'] = sparse_graph.attr_matrix.indices
        data_dict['attr_indptr'] = sparse_graph.attr_matrix.indptr
        data_dict['attr_shape'] = sparse_graph.attr_matrix.shape
    elif isinstance(sparse_graph.attr_matrix, np.ndarray):
        data_dict['attr_matrix'] = sparse_graph.attr_matrix

    if sp.isspmatrix(sparse_graph.labels):
        data_dict['labels_data'] = sparse_graph.labels.data
        data_dict['labels_indices'] = sparse_graph.labels.indices
        data_dict['labels_indptr'] = sparse_graph.labels.indptr
        data_dict['labels_shape'] = sparse_graph.labels.shape
    elif isinstance(sparse_graph.labels, np.ndarray):
        data_dict['labels'] = sparse_graph.labels

    if sparse_graph.node_names is not None:
        data_dict['node_names'] = sparse_graph.node_names

    if sparse_graph.attr_names is not None:
        data_dict['attr_names'] = sparse_graph.attr_names

    if sparse_graph.class_names is not None:
        data_dict['class_names'] = sparse_graph.class_names

    if sparse_graph.metadata is not None:
        data_dict['metadata'] = sparse_graph.metadata

    if not filepath.endswith('.npz'):
        filepath += '.npz'

    np.savez(filepath, **data_dict)



def filter_func(filter_dim, features):
    # make filtering
    mu = np.mean(features[:, filter_dim])
    std = np.std(features[:, filter_dim])
    id1_set = set(np.where(features[:, filter_dim] > mu + 3 * std)[0])
    id2_set = set(np.where(features[:, filter_dim] < mu - 3 * std)[0])
    bad_node_set = id1_set | id2_set
    return bad_node_set






'''
main part
'''


dataset = 'steam'

if dataset == 'steam':
    # construct item graph
    if not os.path.exists(os.path.join(os.getcwd(), dataset, 'freq_item_mat.pkl')):
        itemID_userID_dict = {}
        with open(os.path.join(os.getcwd(), dataset, 'user_product_date.csv'), 'rb') as f:
            lines = f.readlines()[1:]
            for i in tqdm(range(len(lines))):
                line = str(lines[i].strip(), 'utf-8').split('|')
                userID = line[1]
                itemID = line[2]
                if itemID not in itemID_userID_dict:
                    itemID_userID_dict[itemID] = set()
                    itemID_userID_dict[itemID].add(userID)
                else:
                    itemID_userID_dict[itemID].add(userID)
        itemID_unique = list(itemID_userID_dict.keys())
        itemID_Idx_map = pd.Series(data=range(len(itemID_unique)), index=itemID_unique)
        freq_item_mat = np.zeros(shape=[len(itemID_userID_dict), len(itemID_userID_dict)])
        for i in tqdm(range(len(itemID_unique))):
            itemID1 = itemID_unique[i]
            for j in range(i + 1, len(itemID_unique)):
                itemID2 = itemID_unique[j]
                freq_item_mat[itemID_Idx_map[itemID1], itemID_Idx_map[itemID2]] = len(itemID_userID_dict[itemID1] & itemID_userID_dict[itemID2])
                freq_item_mat[itemID_Idx_map[itemID2], itemID_Idx_map[itemID1]] = len(itemID_userID_dict[itemID1] & itemID_userID_dict[itemID2])
        pickle.dump(itemID_Idx_map, open(os.path.join(os.getcwd(), dataset, 'itemID_Idx_map.pkl'), 'wb'))
        pickle.dump(freq_item_mat, open(os.path.join(os.getcwd(), dataset, 'freq_item_mat.pkl'), 'wb'))
    else:
        itemID_Idx_map = pickle.load(open(os.path.join(os.getcwd(), dataset, 'itemID_Idx_map.pkl'), 'rb'))
        freq_item_mat = pickle.load(open(os.path.join(os.getcwd(), dataset, 'freq_item_mat.pkl'), 'rb'))

    # construct sparse feature matrix
    itemID_tagID_dict = {}
    with open(os.path.join(os.getcwd(), dataset, 'product_tags.csv'), 'r') as f:
        lines = f.readlines()[1:]
        for i in tqdm(range(len(lines))):
            line = lines[i].strip().split('|')
            itemID = line[0]
            tagIDSet = set(line[2].strip('[').strip(']').split(', '))
            if itemID in list(itemID_Idx_map.index):
                if itemID not in itemID_tagID_dict:
                    itemID_tagID_dict[itemID] = tagIDSet
                else:
                    itemID_tagID_dict[itemID] = itemID_tagID_dict[itemID] | tagIDSet
            else:
                pass

    all_tags = set()
    for ele in itemID_tagID_dict:
        all_tags = all_tags | itemID_tagID_dict[ele]
    all_tags = list(all_tags)
    tagID_Idx_map = pd.Series(data=range(len(all_tags)), index=all_tags)

    indices = []
    values = []
    for itemID in tqdm(itemID_tagID_dict):
        itemIdx = itemID_Idx_map[itemID]
        tagID_list = list(itemID_tagID_dict[itemID])
        for tagID in tagID_list:
            tagIdx = tagID_Idx_map[tagID]
            indices.append([itemIdx, tagIdx])
            values.append(1.0)
    indices = np.array(indices)
    values = np.array(values)
    sp_fts = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=[len(itemID_Idx_map), len(tagID_Idx_map)])
    pickle.dump(sp_fts, open(os.path.join(os.getcwd(), dataset, 'sp_fts.pkl'), 'wb'))
    pickle.dump(itemID_tagID_dict, open(os.path.join(os.getcwd(), dataset, 'itemID_tagID_dict.pkl'), 'wb'))
    pickle.dump(tagID_Idx_map, open(os.path.join(os.getcwd(), dataset, 'tagID_Idx_map.pkl'), 'wb'))













