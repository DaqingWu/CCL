# -*- encoding: utf-8 -*-

import sys
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

######################################## Load Data ########################################
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dataname):
    '''
    ind.dataname.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataname.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataname.allx => the feature vectors of both labeled and unlabeled training instances (a superset of ind.dataname.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataname.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataname.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataname.ally => the labels for instances in ind.dataname.allx as numpy.ndarray object;
    ind.dataname.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
    ind.dataname.test.index => the indices of test instances in graph, for the inductive setting as list object.
    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open('./data/ind.{}.{}'.format(dataname, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file('./data/ind.{}.test.index'.format(dataname))
    test_idx_sorted = np.sort(test_idx)

    if dataname == 'cite':
        # For citeseer, there are some isolated nodes in the graph
        # Find isolated nodes, add them as zero-vecs
        test_idx_range = range(min(test_idx), max(test_idx)+1)

        tx_extended = sp.lil_matrix((len(test_idx_range), x.shape[1]))
        tx_extended[test_idx_sorted-min(test_idx_sorted), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range), y.shape[1]))
        ty_extended[test_idx_sorted-min(test_idx_sorted), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_sorted.tolist()

    x = torch.FloatTensor(np.array(features.todense()))
    y = torch.LongTensor(np.argmax(labels, -1))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return x, y, adj, idx_train, idx_val, idx_test

######################################## Drop Augmentations ########################################
def aug_node(x, drop_node, training):
    num_node = x.shape[0]
    if training:
        masks = torch.bernoulli(1. - torch.ones(num_node) * drop_node).cuda()
        x = torch.mul(x, masks.unsqueeze(1))
    else:
        x = x*(1. - drop_node)
    return x