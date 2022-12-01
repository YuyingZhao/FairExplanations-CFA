import time
import argparse
import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import scipy.sparse as sp
from utils import load_data_util, fair_metric, seed_everything, feature_norm, normalize_scipy, group_distance
from mlp import *
from torch_geometric.utils import dropout_adj, convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
import matplotlib.pyplot as plt
import pickle
import random
from explanation_metrics import *
from parse import *


def train(epoch, model):
    model.train()
    optimizer.zero_grad()

    hidden, output = model(x=X)
    l_classification = F.cross_entropy(output[idx_train], labels[idx_train].long())
    
    if epoch < args.opt_start_epoch: # warm start
        loss_train = l_classification
    else: # apply distance-based loss
        X_masked, _ = interpretation.generate_masked_X(idx=idx_train.cpu().numpy(), model=model)
        hidden_masked, output_masked = model(x=X_masked)
        l_distance = group_distance(hidden, idx_train.cpu().numpy(), sens.cpu().numpy(), \
            labels.cpu().numpy())
        l_distance_masked = group_distance(hidden_masked, range(len(idx_train)), sens[idx_train].cpu().numpy(), \
            labels[idx_train].cpu().numpy())
        loss_train = l_classification + args.lambda_*(l_distance+l_distance_masked)
    loss_train.backward()
    optimizer.step()

    if epoch >= (args.epochs/2):
        if epoch % 10 == 0:
            model.eval()

            hidden, output = model(x=X)
            preds = (output.argmax(axis=1)).type_as(labels)
           
            # accuracy-related metrics (validation)
            acc_val = accuracy_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            auc_roc_val = roc_auc_score(one_hot_labels[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
            f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            
            # traditional fairness-related metrics (validation)
            sp_val, eo_val = fair_metric(preds[idx_val.cpu().numpy()].cpu().numpy(), labels[idx_val.cpu().numpy()].cpu().numpy(),\
                sens[idx_val.cpu().numpy()].cpu().numpy())

            p0_val, p1_val, REF_val, v0_val, v1_val, VEF_val \
            = interpretation.interprete(model=model, idx=idx_val.cpu().numpy())

            # record validation logs
            logs.append([
                l_classification.item(), l_distance.item(), l_distance_masked.item(), \
                loss_train.item(), \
                acc_val, auc_roc_val, f1_val, \
                sp_val, eo_val, \
                p0_val, p1_val, REF_val, \
                v0_val, v1_val, VEF_val
                ])

if __name__ == '__main__':
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.seed)

    train_log_path = "./train_logs/"+args.dataset
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
        print("The new directory is created for saving the training logs.")

    # load dataset
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(args.dataset)
    adj_ori = adj
    adj = normalize_scipy(adj)
    features = feature_norm(features)
    X = features.float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    one_hot_labels = np.zeros((len(labels), labels.max()+1))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    if args.dataset == "german":
        arch = [X.shape[1], 4, 2]
    else:
        arch = [X.shape[1], 8, 2]

    # create model
    model = MLP(arch, dropout=args.dropout).float()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        torch.cuda.set_device(args.cuda_device)
        model.cuda()
        edge_index.cuda()
        X = X.cuda()
        labels = labels.cuda()
        idx_train = idx_train
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()

    interpretation = Interpreter(features=X, edge_index=edge_index,\
        utility_labels=labels, sensitive_labels=sens, top_ratio=args.top_ratio, \
        topK=args.topK)

    logs = []

    # Train model
    for epoch in tqdm(range(args.epochs)):
        train(epoch, model)

    logs = np.array(logs)
    filename = "./train_logs/"+str(args.dataset)+"/lambda_"+str(args.lambda_)+\
    "_seed_"+str(args.seed)+"_"+str(args.lr)+"_"+str(args.weight_decay)\
    +"_"+str(args.dropout)+".npy"
    np.save(open(filename, 'wb'), logs)

