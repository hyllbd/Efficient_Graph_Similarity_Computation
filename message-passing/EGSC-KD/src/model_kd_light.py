import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool, ConfusionAttentionModule
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs
from trans_modules import CrossAttentionModule

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
import dgl.function as fn

import pdb

from layers import SETensorNetworkModule, AttentionModule_fix 
from layers import SEAttentionModule, repeat_certain_graph

class EGSC_generator(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSC_generator, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()
        # self.node_embeddings = None

    def calculate_bottleneck_features(self):
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins + self.dim_aug_feats
        else:
            self.feature_count = self.args.tensor_neurons * 1 + self.dim_aug_feats

    def setup_layers(self):
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1 * 1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2 * 1, self.args.filters_3)
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
        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        
        if self.args.diffpool: # diffpool = False
            self.attention = DiffPool(self.args)
        else: 
            self.attention = AttentionModule(self.args, self.args.filters_3)
            self.attention_level2 = AttentionModule(self.args, self.args.filters_2 * self.scaler_dim)
            self.attention_level1 = AttentionModule(self.args, self.args.filters_1 * self.scaler_dim)

    def convolutional_pass_level1(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.args.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features):
        features_2 = self.convolution_2(features, edge_index)
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.args.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features):
        features_out = self.convolution_3(features, edge_index)
        return features_out

    def forward(self, edge_index, features, batch):

        features_level1 = self.convolutional_pass_level1(edge_index, features)
        features_level2 = self.convolutional_pass_level2(edge_index, features_level1)
 
        pooled_features_level1 = self.attention_level1(features_level1, batch) 
        
        pooled_features_level2 = self.attention_level2(features_level2, batch) 

        pooled_features_all = torch.cat((pooled_features_level2,pooled_features_level1),dim=1)
        
        # self.node_embeddings = pooled_features_all
        # print(f'[light] pooled_features_all.shape: {pooled_features_all.shape}') # torch.Size([128, 96])
        return  pooled_features_all

class EGSC_fusion(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSC_fusion, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()
        # self.node_embeddings = None

    def setup_layers(self):
        # self.filter_dim_all = self.args.filters_3 + self.args.filters_2 + self.args.filters_1
        self.filter_dim_all = self.args.filters_2 + self.args.filters_1 # filters_1=64, filters_2=32 64+32=96
        self.score_attention = SEAttentionModule(self.args, self.filter_dim_all * 2) # self.filter_dim_all * 2 = 96 * 2 = 192
        self.feat_layer = torch.nn.Linear(self.filter_dim_all * 2, self.filter_dim_all)
        self.fully_connected_first = torch.nn.Linear(self.filter_dim_all, self.args.bottle_neck_neurons)
        
    def forward(self, pooled_features_1_all, pooled_features_2_all):
        scores = torch.cat((pooled_features_1_all,pooled_features_2_all),dim=1) 
        
        
        scores = self.feat_layer(self.score_attention(scores) + scores)  
        scores = F.relu(self.fully_connected_first(scores)) 
        return  scores 

class EGSC_fusion_classifier(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSC_fusion_classifier, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def setup_layers(self):
        self.feat_layer = torch.nn.Linear(self.args.bottle_neck_neurons * 2, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def forward(self, scores):
        scores = F.relu(self.feat_layer(scores))
        scores = torch.sigmoid(self.scoring_layer(scores)).view(-1) # dim of score: 128 * 0
        return  scores 

class EGSC_classifier(torch.nn.Module): 
    def __init__(self, args, number_of_labels):
        super(EGSC_classifier, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def setup_layers(self):
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1) 

    def forward(self, scores):
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        return  score 


class EGSC_teacher(torch.nn.Module): 
    def __init__(self, args, number_of_labels):
        super(EGSC_teacher, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def calculate_bottleneck_features(self):
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins + self.dim_aug_feats
        else:
            self.feature_count = (self.args.filters_1 + self.args.filters_2 + self.args.filters_3 ) // 2
            
            # light
            self.feature_count_light = (self.args.filters_1 + self.args.filters_2) // 2

    def setup_layers(self):
        self.calculate_bottleneck_features()
        if self.args.gnn_operator_fix == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        elif self.args.gnn_operator_fix == 'gin':
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
        elif self.args.gnn_operator_fix == 'gat':
            self.convolution_1 = GATConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator_fix == 'sage':
            self.convolution_1 = SAGEConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3)

        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        
        if self.args.diffpool:
            self.attention = DiffPool(self.args)
        else:
            self.attention_level3 = AttentionModule_fix(self.args, self.args.filters_3)
            self.attention_level2 = AttentionModule_fix(self.args, self.args.filters_2 * self.scaler_dim)
            self.attention_level1 = AttentionModule_fix(self.args, self.args.filters_1 * self.scaler_dim)

        self.cross_attention_level2 = CrossAttentionModule(self.args, self.args.filters_2)
        self.cross_attention_level3 = CrossAttentionModule(self.args, self.args.filters_3)
        self.cross_attention_level4 = CrossAttentionModule(self.args, self.args.filters_4)

        self.tensor_network_level3 = SETensorNetworkModule(self.args,dim_size=self.args.filters_3 * self.scaler_dim)
        self.tensor_network_level2 = SETensorNetworkModule(self.args,dim_size=self.args.filters_2 * self.scaler_dim)
        self.tensor_network_level1 = SETensorNetworkModule(self.args,dim_size=self.args.filters_1 * self.scaler_dim)
        
        self.fully_connected_first_light = torch.nn.Linear(self.feature_count_light, self.args.bottle_neck_neurons)
        
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
       
        # self.score_attention = SEAttentionModule(self.args, self.feature_count)
        # light
        self.score_attention_light = SEAttentionModule(self.args, self.feature_count_light) 
        
    def convolutional_pass_level1(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.args.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features):
        features_2 = self.convolution_2(features, edge_index)
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.args.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features):
        features_3 = self.convolution_3(features, edge_index)
        features_3 = F.relu(features_3)
        features_3 = F.dropout(features_3, p=self.args.dropout, training=self.training)
        return features_3

    def convolutional_pass_level4(self, edge_index, features):
        features_out = self.convolution_4(features, edge_index)
        return features_out
        

    def forward(self, edge_index_1, features_1, batch_1, edge_index_2, features_2, batch_2):

        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2)

        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2) 

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2)
        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) 
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2) 

        # features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1)
        # features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2)
        # pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        # pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16

        # scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        scores = torch.cat((scores_level2, scores_level1), dim=1)
        # print(f'[light] EGSC_teacher scores.shape: {scores.shape}')

        # scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))
        # light
        scores = F.relu(self.fully_connected_first_light(self.score_attention_light(scores)*scores + scores))
        
        return  scores

class logits_D(torch.nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = torch.nn.Linear(self.n_hidden, self.n_hidden) # torch.nn.Linear(in_features, out_features, bias=True) 
        self.relu = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid() 
        self.lin2 = torch.nn.Linear(self.n_hidden, self.n_class+1, bias=False) 

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        if self.n_class == 16:
            out = self.relu(out)
        # elif self.n_class == 1:
        #     out = self.sigmoid(out)
        dist = self.lin2(out)
        return dist

class local_emb_D(torch.nn.Module): 
    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden 
        self.d = torch.nn.Parameter(torch.ones(size=(n_hidden, ))) 
        self.scale = torch.nn.Parameter(torch.full(size=(1, ), fill_value= 0.5)) 

    def forward(self, emb, g):
        emb = F.normalize(emb, p=2)
        
        g.ndata['e'] = emb 
        g.ndata['ew'] = emb @ torch.diag(self.d) 
        g.apply_edges(fn.u_dot_v('ew', 'e', 'z')) 
        pair_dis = g.edata['z'] 
        return pair_dis * self.scale

class global_emb_D(torch.nn.Module): 
    def __init__(self, n_hidden): 
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden # 16
        self.d = torch.nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = torch.nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, summary):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d) 
        assert summary.shape[-1] == 1 
        sim = sim @ summary 
        return sim * self.scale