# -*- encoding: utf-8 -*-

import os
import argparse
from loguru import logger
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from utils import load_data, sparse_mx_to_torch_sparse_tensor
from utils import aug_node
from model import CCL
    
if __name__ == "__main__":
    if os.path.isdir('./result/CCL/') == False:
        os.makedirs('./result/CCL/')
    if os.path.isdir('./log/CCL/') == False:
        os.makedirs('./log/CCL/')

    parser = argparse.ArgumentParser(description='CCL')
    parser.add_argument('--dataname', default='cite', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=100, type=int)

    parser.add_argument('--dim_hidden', default=32, type=list)
    parser.add_argument('--num_class', default=6, type=int)
    parser.add_argument('--dropout_gcn', default=0, type=float)
    parser.add_argument('--dropout_pre', default=0, type=float)
    parser.add_argument('--num_hop', default=2, type=int)

    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--patience', default=300, type=int)

    parser.add_argument('--augs', default=2, type=int)
    parser.add_argument('--drop_node', default=0.5, type=float)

    parser.add_argument('--lambdas', default=0.3, type=float)
    parser.add_argument('--mu', default=0.3, type=float)
    parser.add_argument('--t_c', default=0.3, type=float)
    parser.add_argument('--t_p', default=1, type=float)
    
    parser.add_argument('--use_bn_gcn', default=False, type=bool)
    parser.add_argument('--use_bn_pre', default=False, type=bool)

    parser.add_argument('--model_save_path', default='pkl', type=str)
    args = parser.parse_args()
    args.model_save_path = './result/CCL/{}_model.pkl'.format(args.dataname)

    if args.dataname == 'cora':
        args.num_class = 7
        args.dropout_gcn = 0.5
        args.dropout_pre = 0
        args.num_hop = 8
        args.lambdas = 0.7
        args.mu = 0.5
        args.t_c = 0.5
        args.t_p = 0.1

    if args.dataname == 'pubm':
        args.num_class = 3
        args.learning_rate = 0.1
        args.weight_decay = 1e-3
        args.dropout_gcn = 0.5
        args.dropout_pre = 0.5
        args.num_hop = 5
        args.patience = 100
        args.lambdas = 0.3
        args.mu = 1
        args.t_c = 1
        args.t_p = 0.1
        args.use_bn_gcn = True
        args.use_bn_pre = True

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.add('./log/CCL/{}.log'.format(args.dataname), rotation="500 MB", level="INFO")
    logger.info(args)

    x, y, adj, idx_train, idx_val, idx_test = load_data(args.dataname)
    x = torch.nn.functional.normalize(x, p=1, dim=1)

    # add self-loop
    adj_self_loop = adj + sp.eye(adj.shape[0])
    # D^-0.5*A*D^-0.5
    d1 = np.array(adj_self_loop.sum(axis=1))**(-0.5)
    d2 = np.array(adj_self_loop.sum(axis=0))**(-0.5)
    d1 = sp.diags(d1[:,0], format='csr')
    d2 = sp.diags(d2[0,:], format='csr')
    adj_norm = adj_self_loop.dot(d1)
    adj_norm = d2.dot(adj_norm)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    Model = CCL(dim_input=x.shape[1], \
                  dim_hidden=args.dim_hidden, \
                  dim_output=args.num_class, \
                  num_hop=args.num_hop, \
                  dropout_gcn=args.dropout_gcn, \
                  dropout_pre=args.dropout_pre, \
                  use_bn_gcn=args.use_bn_gcn, \
                  use_bn_pre=args.use_bn_pre, \
                  bias_gcn=True, bias_pre=True).cuda()
    optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    acc_list = []
    bad_counter = 0
    acc_max = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs+1):
        Model.train()
        x_list = []
        for s in range(args.augs):
            x_list.append(aug_node(x.cuda(), args.drop_node, training=True))

        h_list = []
        log_prob_list = []
        for s in range(args.augs):
            h, log_prob = Model.forward(x_list[s], adj_norm.cuda(), args.t_p)
            h_list.append(h)
            log_prob_list.append(log_prob)
        
        ######################################## Supervised Loss ########################################
        loss_sup = 0.
        for s in range(args.augs):
            loss_sup += torch.nn.functional.nll_loss(log_prob_list[s][idx_train], y[idx_train].cuda())
        loss_sup = loss_sup/args.augs
        # print(loss_sup)
        ######################################## Center-Level Loss ########################################
        centers_list = []
        for s in range(args.augs):
            centers_list.append(torch.nn.functional.normalize(torch.sum(torch.mul(h_list[s].repeat(args.num_class, 1, 1), torch.exp(log_prob_list[s]).T.unsqueeze(2)), dim=1), p=2, dim=1))
        
        score_11 = torch.exp(torch.mm(centers_list[0], centers_list[0].T)/args.t_c)
        score_22 = torch.exp(torch.mm(centers_list[1], centers_list[1].T)/args.t_c)
        score_12 = torch.exp(torch.mm(centers_list[0], centers_list[1].T)/args.t_c)
        score_21 = torch.exp(torch.mm(centers_list[1], centers_list[0].T)/args.t_c)
        
        loss_cen = 0.
        for c in range(args.num_class):
            loss_cen += torch.log(score_12[c][c] / (torch.sum(score_11[c])-score_11[c][c] + torch.sum(score_22[c])-score_22[c][c] + torch.sum(score_12[c])-score_12[c][c] + torch.sum(score_21[c])-score_21[c][c]))
        loss_cen = - loss_cen/args.num_class
        # print(loss_cen)
        ######################################## Instance-Level Loss ########################################
        loss_ins = torch.mean(torch.sum(torch.pow(torch.exp(log_prob_list[0]) - torch.exp(log_prob_list[1]), 2), dim=1))
        # print(loss_ins)
        loss = loss_sup + args.lambdas * loss_cen + args.mu * loss_ins

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Model.eval()
            x_ = aug_node(x.cuda(), args.drop_node, training=False)
            h, log_prob = Model.forward(x_, adj_norm.cuda(), args.t_p)
            loss_train = torch.nn.functional.nll_loss(log_prob[idx_train], y[idx_train].cuda())
            acc_train = accuracy_score(y[idx_train], torch.exp(log_prob[idx_train]).cpu().numpy().argmax(1))
            logger.info('Epoch {}/{} Train loss: {:.4f} ACC: {:.4f} Bad Counter {}/{}'.format(epoch, args.epochs, loss_train.cpu().item(), acc_train, bad_counter, args.patience))
            loss_val = torch.nn.functional.nll_loss(log_prob[idx_val], y[idx_val].cuda())
            acc_val = accuracy_score(y[idx_val], torch.exp(log_prob[idx_val]).cpu().numpy().argmax(1))

            acc_list.append(acc_val)

        if acc_val >= acc_max:
            acc_max = max(acc_list)
            best_epoch = epoch
            torch.save(Model.state_dict(), args.model_save_path)
            logger.info('Epoch {}/{} Val loss: {:.4f} ACC: {:.4f}'.format(epoch, args.epochs, loss_val.cpu().item(), acc_val))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            logger.info('Early stop! Epoch {}/{} ACC: {:.4f}'.format(best_epoch, args.epochs, acc_max))
            break

    logger.info('Loading {} th epoch'.format(best_epoch))
    Model.load_state_dict(torch.load(args.model_save_path))

    with torch.no_grad():
        Model.eval()
        x_ = aug_node(x.cuda(), args.drop_node, training=False)
        h, log_prob = Model.forward(x_, adj_norm.cuda(), args.t_p)
        loss_test = torch.nn.functional.nll_loss(log_prob[idx_test], y[idx_test].cuda())
        acc_test = accuracy_score(y[idx_test], torch.exp(log_prob[idx_test]).cpu().numpy().argmax(1))
        logger.info('Test loss: {:.4f} ACC: {:.4f}'.format(loss_test.cpu().item(), acc_test))
