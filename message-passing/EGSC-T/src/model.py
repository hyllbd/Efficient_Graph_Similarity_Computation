import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import DiffPool, ConfusionAttentionModule
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import pdb
from layers import AttentionModule as AttentionModule
from layers import SETensorNetworkModule as TensorNetworkModule
from layers import SEAttentionModule
from mpnn import SMPModel
from ginskip import GINWithSkip

class EGSCT_generator(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSCT_generator, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = (self.args.filters_1 + self.args.filters_2 + self.args.filters_3 ) // 2

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1) # self.number_of_labels = self.training_graphs.num_features, i.e. the number of node features
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))
            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        elif self.args.gnn_operator == 'gat':
            self.convolution_1 = GATConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'sage':
            self.convolution_1 = SAGEConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'mpnn':
            pass # implemented in convolutional_pass_level1 2 3
        elif self.args.gnn_operator == 'ginmp':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
        elif self.args.gnn_operator == 'ginskip':
            self.convolution_1 = GINWithSkip(self.number_labels, self.args.filters_1)
            self.convolution_2 = GINWithSkip(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GINWithSkip(self.args.filters_2, self.args.filters_3)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.attention_level3 = AttentionModule(self.args, self.args.filters_3)

        self.attention_level2 = AttentionModule(self.args, self.args.filters_2 * self.scaler_dim)

        self.attention_level1 = AttentionModule(self.args, self.args.filters_1 * self.scaler_dim)

        self.tensor_network_level3 = TensorNetworkModule(self.args,dim_size=self.args.filters_3 * self.scaler_dim)
        self.tensor_network_level2 = TensorNetworkModule(self.args,dim_size=self.args.filters_2 * self.scaler_dim)
        self.tensor_network_level1 = TensorNetworkModule(self.args,dim_size=self.args.filters_1 * self.scaler_dim)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

        self.score_attention = SEAttentionModule(self.args, self.feature_count)

    def setup_mpnn(self, num_node_features, hidden_final):
        # print('[EGSC-T/src/model.py] setup_mpnn num_node_features: ', num_node_features)
        # num_input_features = num_node_features + 1
        smp_model = SMPModel(num_input_features=(num_node_features+1), num_edge_features=1, num_classes=1, num_layers=1,
                            hidden=32, residual=False, use_edge_features=False, shared_extractor=True,
                            hidden_final=hidden_final, use_batch_norm=True, use_x=False, map_x_to_u=True,
                            num_towers=8, simplified=False, cuda_id=self.args.cuda_id)
        return smp_model
        
    def convolutional_pass_level1(self, edge_index, features, batch, i):
        """
        Making convolutional pass.
        """
        # gin: features.shape: [num_nodes, num_node_features] e.g. [571, 29]  # smp: features.shape: [num_nodes, num_node_features] e.g. [571, 29] 
        
        if self.args.gnn_operator == 'mpnn':
            self.convolution_1 = self.setup_mpnn(num_node_features=features.shape[1], hidden_final=self.args.filters_1)
            features = self.convolution_1(edge_index, features, batch, i)
        else:
            features = self.convolution_1(features, edge_index) 
        
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.args.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features, batch, i):
        if self.args.gnn_operator == 'mpnn':
            self.convolution_2 = self.setup_mpnn(num_node_features=features.shape[1], hidden_final=self.args.filters_2)
            features_2 = self.convolution_2(edge_index, features, batch, i)
        else:
            features_2 = self.convolution_2(features, edge_index) 
        
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.args.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features, batch, i):
        if self.args.gnn_operator == 'mpnn' or self.args.gnn_operator == 'ginmp':
            self.convolution_3 = self.setup_mpnn(num_node_features=features.shape[1], hidden_final=self.args.filters_3)
            features_3 = self.convolution_3(edge_index, features, batch, i)
        else:
            features_3 = self.convolution_3(features, edge_index)
            
        features_3 = F.relu(features_3)
        features_3 = F.dropout(features_3, p=self.args.dropout, training=self.training)
        return features_3
        
    def forward(self, edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2):
        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1, batch_1, i_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2, batch_2, i_2)
        
        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1, batch_1, i_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2, batch_2, i_2)

        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) # 128 * 32
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1, batch_1, i_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2, batch_2, i_2)
        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)
        
        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        
        scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))

        return  scores 

class EGSCT_classifier(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSCT_classifier, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def forward(self, scores):
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1) # dim of score: 128 * 0
        return  score 