import time
import argparse
import numpy as np
import torch
import sys
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

def train(epoch, model, max_score):
    model.train()
    optimizer.zero_grad()

    hidden, output = model(x=X)
    l_classification = F.cross_entropy(output[idx_train], labels[idx_train].long())
   
    if epoch < args.opt_start_epoch:
        loss_train = l_classification
    else:
        # distance-based loss
        X_masked, _ = interpretation.generate_masked_X(idx=idx_train.cpu().numpy(), model=model)
        hidden_masked, output_masked = model(x=X_masked)
        l_distance = group_distance(hidden, idx_train.cpu().numpy(), sens.cpu().numpy(), \
            labels.cpu().numpy())
        l_distance_masked = group_distance(hidden_masked, range(len(idx_train)), sens[idx_train].cpu().numpy(),\
         labels[idx_train].cpu().numpy())
        loss_train = 1*l_classification + args.lambda_*(l_distance+l_distance_masked)
    loss_train.backward()
    optimizer.step()

    if epoch >= (args.epochs/2):
        if epoch % 10 == 0:
            model.eval()

            hidden, output = model(x=X)
            preds = (output.argmax(axis=1)).type_as(labels)

            # utility metrics (validation)
            acc_val = accuracy_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            auc_roc_val = roc_auc_score(one_hot_labels[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
            f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            
            # traditional fairness metrics (validation)
            sp_val, eo_val = fair_metric(preds[idx_val.cpu().numpy()].cpu().numpy(), labels[idx_val.cpu().numpy()].cpu().numpy(),\
                sens[idx_val.cpu().numpy()].cpu().numpy())

            # explanation fairness metrics (validation)
            p0_val, p1_val, REF_val, v0_val, v1_val, VEF_val \
            = interpretation.interprete(model=model, idx=idx_val.cpu().numpy())

            # record the best score
            score = (auc_roc_val+f1_val+acc_val)/3.0-(sp_val+eo_val)/2.0\
            -(REF_val+VEF_val)/2.0
            if score >= max_score:
                max_score = score
                torch.save(model.state_dict(), "./weights/"+args.dataset+\
                    "_"+str(args.seed)+"_best.npy")            
    return max_score

if __name__ == '__main__':
    # set args
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.seed)

    if args.dataset == "german":
        best_hyper_parameters = [[0.01, 0.001, 0.5, 10.0], \
        [0.01, 1e-05, 0.3, 10.0], [0.01, 0.001, 0.1, 10.0], \
        [0.01, 0.0001, 0.5, 10.0], [0.01, 0.0001, 0.3, 10.0]]
    elif args.dataset == "bail":
        best_hyper_parameters = [[0.01, 1e-05, 0.5, 0.01], \
        [0.01, 0.001, 0.3, 0.01], [0.01, 0.001, 0.1, 0.001],\
        [0.01, 0.001, 0.3, 0.01], [0.01, 0.001, 0.3, 0.01]]
    elif args.dataset == "math":
        best_hyper_parameters = [[0.01, 0.001, 0.5, 0.1], \
        [0.01, 0.001, 0.5, 0.1], [0.01, 0.001, 0.1, 0.1], \
        [0.01, 0.0001, 0.5, 0.01], [0.01, 1e-05, 0.5, 0.1]]
    elif args.dataset == "por":
        best_hyper_parameters = [[0.01, 0.0001, 0.1, 0.01], \
        [0.01, 0.0001, 0.3, 0.01], [0.01, 0.0001, 0.3, 0.001], \
        [0.01, 0.0001, 0.1, 0.01], [0.01, 0.001, 0.3, 0.001]]

    args.lr = best_hyper_parameters[args.seed-1][0]
    args.weight_decay = best_hyper_parameters[args.seed-1][1]
    args.dropout = best_hyper_parameters[args.seed-1][2]
    args.lambda_ = best_hyper_parameters[args.seed-1][3]

    # load dataset 
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(args.dataset)
    adj_ori = adj
    adj = normalize_scipy(adj)
    features = feature_norm(features)
    X = features.float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    one_hot_labels = np.zeros((len(labels), labels.max()+1))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    # load model
    if args.dataset == "german":
        arch = [X.shape[1], 4, 2]
    else:
        arch = [X.shape[1], 8, 2]
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
    max_score = -100
    for epoch in tqdm(range(args.epochs)):
        max_score = train(epoch, model, max_score)

    # load best model
    model = MLP(arch, dropout=args.dropout).float().cuda()
    model.load_state_dict(torch.load("./weights/"+args.dataset+"_"+str(args.seed)+\
        "_best.npy"))
    model.eval()

    # evaluate the performance in test dataset
    hidden, output = model(x=X)
    preds = (output.argmax(axis=1)).type_as(labels)
    # utility metrics (test)
    acc_test = accuracy_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
    auc_roc_test = roc_auc_score(one_hot_labels[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
    f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())

    # traditional fairness metrics (test)
    sp_test, eo_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(), labels[idx_test.cpu().numpy()].cpu().numpy(),\
     sens[idx_test.cpu().numpy()].cpu().numpy())

    # explanation fairness metrics (test)
    p0_test, p1_test, REF_test, v0_test, v1_test, VEF_test \
    = interpretation.interprete(model=model, \
        idx=idx_test.cpu().numpy())
    logs = [
        acc_test, auc_roc_test, f1_test, \
        sp_test, eo_test, \
        p0_test, p1_test, REF_test, \
        v0_test, v1_test, VEF_test
        ]

    # write logs to file
    logs = np.array(logs)
    filename = "./test_logs/"+str(args.dataset)+"_best_test_result.txt"
    f = open(filename, "a+")
    logs_string = ' '.join([str(t) for t in logs])
    logs_string += "\n"
    f.write(logs_string)


