from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import time
import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score,precision_recall_curve,auc
import argparse
from optimizer import  OptimizerCAN
from input_data import load_AN, load_my_data
from model import CAN
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_feas
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

# Settings

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=750,
                    help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset string.')
args = parser.parse_args(args=[])
dataset_str = args.dataset

# Load data
# adj, features = load_AN(dataset_str)
adj, features, sim= load_my_data()
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

adj = adj_train
features_orig = features
features = sp.lil_matrix(features)
link_predic_result_file = "result/AGAE_{}.res".format(dataset_str)
embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(dataset_str)
embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(dataset_str)
embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(dataset_str)
embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(dataset_str)

adj_norm = preprocess_graph(adj)
sim_norm = preprocess_graph(sim)


num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
# Create model
#args can be one parameter
#创建model这里的参数是传到CAN的构造函数(init)处
model = CAN(args.hidden1,args.hidden2,num_features, num_nodes, features_nonzero,args.dropout).to(device)
pos_weight_u = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm_u = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
pos_weight_a = float(features[2][0] * features[2][1] - len(features[1])) / len(features[1])
norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)
# Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                       
cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg,preds_sub_u):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    # adj_rec = sess.run(model.reconstructions[0], feed_dict=feed_dict).reshape([num_nodes, num_nodes])
    if(use_gpu):
        adj_rec=preds_sub_u.view(num_nodes,num_nodes).cpu().data.numpy()
    else:
        adj_rec=preds_sub_u.view(num_nodes,num_nodes).data.numpy()
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    # print(np.min(adj_rec))
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg]).astype(np.float)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    # f1 = f1_score(labels_all, labels_sub_u)
    m_pred = torch.ge(torch.Tensor(preds_all),0.5)
    f1 = f1_score(labels_all,m_pred)
    precision, recall, thresholds = precision_recall_curve(labels_all,preds_all)
    pr_auc = auc(recall,precision)
    print('u_f1:',f1,'pr_auc:',pr_auc)
    return roc_score, ap_score, f1, pr_auc

def get_roc_score_a(feas_pos, feas_neg,preds_sub_a):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    # fea_rec = sess.run(model.reconstructions[1], feed_dict=feed_dict).reshape([num_nodes, num_features])
    if(use_gpu):
        fea_rec = preds_sub_a.view(num_nodes, num_features).cpu().data.numpy()
    else:
        fea_rec = preds_sub_a.view(num_nodes, num_features).data.numpy()
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    logits=logits.clamp(-10,10)
    return targets * -torch.log(torch.sigmoid(logits)) *pos_weight + (1 - targets) * -torch.log(1 - torch.sigmoid(logits))

cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
features_label = sparse_to_tuple(features_orig)
'''sparse_to_tuple返回三个参数，第1个的index，即指明哪一行哪一列有指，第2个就是具体的value，第三个是矩阵的维度
这三个参数刚好是torch.sparse将稠密矩阵转成稀疏矩阵的参数，具体可以查官网文档'''
features=torch.sparse.FloatTensor(torch.LongTensor(features[0]).t(),torch.FloatTensor(features[1]),features[2]).to(device)
adj_norm=torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(np.int32)).t(), torch.FloatTensor(adj_norm[1]), adj_norm[2]).to(device)
sim_norm=torch.sparse.FloatTensor(torch.LongTensor(sim_norm[0].astype(np.int32)).t(), torch.FloatTensor(sim_norm[1]), sim_norm[2]).to(device)
# Train model
for epoch in range(args.epochs):
    model.train()
    t = time.time()
    optimizer.zero_grad()
    #Get result
    '''train model这里参数传到CAN的forward处，features即论文模型图中的X矩阵(结点属性矩阵)，adj_norm是模型中的A矩阵，即邻接矩阵'''
    preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std=model(features,adj_norm,sim_norm)

    labels_sub_u=torch.from_numpy(adj_orig.toarray()).flatten().float().to(device)
    labels_sub_a=torch.from_numpy(features_orig.toarray()).flatten().float().to(device)
    cost_u = norm_u * torch.mean(weighted_cross_entropy_with_logits(logits=preds_sub_u, targets=labels_sub_u, pos_weight=pos_weight_u))
    cost_a = norm_a * torch.mean(weighted_cross_entropy_with_logits(logits=preds_sub_a, targets=labels_sub_a, pos_weight=pos_weight_a))
    
    # Latent loss
    log_lik = cost_u + cost_a
    kl_u = (0.5 ) * torch.mean((1 + 2 * z_u_log_std - z_u_mean.pow(2) - (torch.exp(2*z_u_log_std))).sum(1))/ num_nodes
    kl_a = (0.5 ) * torch.mean((1 + 2 * z_a_log_std - (z_a_mean.pow(2)) -(torch.exp(2*z_a_log_std))).sum(1))/ num_features
    kl = kl_u + kl_a  
    cost = log_lik - kl
    correct_prediction_u = torch.sum(torch.eq(torch.ge(torch.sigmoid(preds_sub_u), 0.5).float(),labels_sub_u).float())/len(labels_sub_u)
    correct_prediction_a = torch.sum(torch.eq(torch.ge(torch.sigmoid(preds_sub_a), 0.5).float(),labels_sub_a).float())/len(labels_sub_a)
    accuracy = torch.mean(correct_prediction_u + correct_prediction_a)
    # print("acc_u:",correct_prediction_u)
    # Compute average loss
    avg_cost = cost
    avg_accuracy = accuracy
    log_lik = log_lik
    kl = kl
    roc_curr, ap_curr, f1, pr_auc= get_roc_score(val_edges, val_edges_false,preds_sub_u)
    roc_curr_a, ap_curr_a = get_roc_score_a(val_feas, val_feas_false,preds_sub_a)
    val_roc_score.append(roc_curr)

    #Run backward
    avg_cost.backward()
    optimizer.step()

    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(avg_cost),
          "log_lik=", "{:.5f}".format(log_lik),
          "KL=", "{:.5f}".format(kl),
          "train_acc=", "{:.5f}".format(avg_accuracy),
          "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_edge_ap=", "{:.5f}".format(ap_curr),
          # "val_edge_ap=", "{:.5f}".format(f1),
          "val_attr_roc=", "{:.5f}".format(roc_curr_a),
          "val_attr_ap=", "{:.5f}".format(ap_curr_a),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")


preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std=model(features,adj_norm,sim_norm)
zll_result = torch.sigmoid(preds_sub_u).reshape((1322, 1322))
zll_result_numpy = zll_result.cpu().detach().numpy()
np.savetxt('zll_result_numpy.txt', zll_result_numpy, fmt='%.2f', delimiter=' ')
roc_score, ap_score, f1, pr_auc= get_roc_score(test_edges, test_edges_false,preds_sub_u)
roc_score_a, ap_score_a = get_roc_score_a(test_feas, test_feas_false,preds_sub_a)

if use_gpu:
    z_u_mean = z_u_mean.cpu()
    z_a_mean = z_a_mean.cpu()
    z_u_log_std = z_u_log_std.cpu()
    z_a_log_std = z_a_log_std.cpu()

np.save(embedding_node_mean_result_file, z_u_mean.data.numpy())
np.save(embedding_attr_mean_result_file, z_a_mean.data.numpy())
np.save(embedding_node_var_result_file, z_u_log_std.data.numpy())
np.save(embedding_attr_var_result_file, z_a_log_std.data.numpy())    
print('Test edge ROC score: ' + str(roc_score))
print('Test edge AP score: ' + str(ap_score))
print('Test edge F1 score: ' + str(f1))
print('Test edge pr_auc score: ' + str(pr_auc))
print('Test attr ROC score: ' + str(roc_score_a))
print('Test attr AP score: ' + str(ap_score_a))
