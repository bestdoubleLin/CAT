import numpy as np
import scipy.sparse as sp


def load_AN(dataset):
    edge_file = open(r"data/{}.edge".format(dataset), 'r')
    attri_file = open(r"data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))
        
    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    return adj, attribute

def load_my_data():
    adj = np.loadtxt('data/DDI.txt')
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    print(adj.shape)
    adj = sp.csc_matrix((np.ones(len(adj_row)), (adj_row, adj_col)), shape=(adj.shape[0], adj.shape[0]))
    attribute = np.loadtxt('data/drugECFPs.csv', delimiter=",")
    print(attribute.shape)
    att_row = attribute.nonzero()[0]
    att_col = attribute.nonzero()[1]
    attribute = sp.csc_matrix((np.ones(len(att_col)), (att_row, att_col)), shape=(attribute.shape[0], attribute.shape[1]))
    sim = np.loadtxt('data/DDSimilarity.txt')
    sim_row = adj.nonzero()[0]
    sim_col = adj.nonzero()[1]
    sim = sp.csc_matrix((np.ones(len(sim_row)), (sim_row, sim_col)), shape=(sim.shape[0], sim.shape[0]))
    # sim=np.zeros((1704,1704))
    return adj,attribute, sim