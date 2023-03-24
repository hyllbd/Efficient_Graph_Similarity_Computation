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
        # print(f'[EGSC-T/src/model.py] 卷积层1 输入 features.shape: {features.shape} edge_index.shape: {edge_index.shape}') 
        # gin: features.shape: [num_nodes, num_node_features] e.g. [571, 29]  # smp: features.shape: [num_nodes, num_node_features] e.g. [571, 29] 
        
        if self.args.gnn_operator == 'mpnn':
            # print(f'[EGSC-T/src/model.py] mpnn卷积层1 输入 data: {data}') # data: DataBatch(edge_index=[2, 1160], i=[64], x=[587, 29], num_nodes=587, batch=[587], ptr=[65])
            self.convolution_1 = self.setup_mpnn(num_node_features=features.shape[1], hidden_final=self.args.filters_1)
            features = self.convolution_1(edge_index, features, batch, i)
        else:
            features = self.convolution_1(features, edge_index) 
        
        # print(f'[EGSC-T/src/model.py] 卷积层1 输出 features.shape: {features.shape}') 
        # gin: features.shape: [num_nodes, self.args.filters_1] e.g. [571, 64]
        
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.args.dropout, training=self.training)
        # print(f'[EGSC-T/src/model.py] 卷积层1 final features_1.shape: {features_1.shape}')
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

    # def convolutional_pass_level4(self, edge_index, features, batch, i):
    #     if self.args.gnn_operator == 'mpnn':
    #         self.convolution_4 = self.setup_mpnn(num_node_features=features.shape[1], hidden_final=self.args.filters_4)
    #         features_out = self.convolution_4(edge_index, features, batch, i)
    #     else:
    #         features_out = self.convolution_4(features, edge_index)
    #     return features_out
        
    def forward(self, edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2):
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，输入的input data:{data}')
        # print(f"[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，data[target].shape为{data['target'].shape}, data[target_ged].shape为{data['target_ged'].shape}")
        
        # if use def forward(self, data):
        # edge_index_1 = data["g1"].edge_index
        # edge_index_2 = data["g2"].edge_index
        # features_1 = data["g1"].x # 读取图1的节点特征
        # features_2 = data["g2"].x # 读取图2的节点特征
        # batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        # batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，data["g1"].batch:{data["g1"].batch}, \ndata["g2"].batch:{data["g2"].batch}, \ndata["g1"].ptr:{data["g1"].ptr}, \ndata["g2"].ptr:{data["g2"].ptr}, \ndata["g1"].i:{data["g1"].i}, \ndata["g2"].i:{data["g2"].i}')
        # torch.set_printoptions(threshold=torch.inf)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，data["g1"].batch:{data["g1"].batch}, \ndata["g2"].batch:{data["g2"].batch}')
        # torch.set_printoptions(profile='default')
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，g1的batch长度为{len(data["g1"].batch)}, g2的batch长度为{len(data["g2"].batch)}')
        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1, batch_1, i_1) # 边索引，节点特征, batch, i(图的编号)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2, batch_2, i_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数, features_level1_1.shape为{features_level1_1.shape}, features_level1_2.shape为{features_level1_2.shape}')
        
        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level1_1.shape为{pooled_features_level1_1.shape}, pooled_features_level1_2.shape为{pooled_features_level1_2.shape}, scores_level1.shape为{scores_level1.shape}')

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1, batch_1, i_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2, batch_2, i_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，features_level2_1.shape为{features_level2_1.shape}, features_level2_2.shape为{features_level2_2.shape}')

        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) # 128 * 32
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level2_1.shape为{pooled_features_level2_1.shape}, pooled_features_level2_2.shape为{pooled_features_level2_2.shape}, scores_level2.shape为{scores_level2.shape}')

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1, batch_1, i_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2, batch_2, i_2)
        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level3_1.shape为{pooled_features_level3_1.shape}, pooled_features_level3_2.shape为{pooled_features_level3_2.shape}, scores_level3.shape为{scores_level3.shape}')
        
        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，scores.shape为{scores.shape}')
        
        scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，最终返回的scores.shape={scores.shape}')

        """ dataset AIDS700nef 打印信息汇总
        输入的data为{
                    'g1': DataBatch(
                        edge_index=[2, 2252], i=[128], x=[1133, 29],
                        num_nodes=1133, batch=[1133], ptr=[129]),
                    'g2': DataBatch(
                        edge_index=[2, 2224], i=[128], x=[1121, 29],
                        num_nodes=1121, batch=[1121], ptr=[129]),
                    'target': tensor([0.4966, 0.4066, 0.531...])  # data[target].shape为torch.Size([128])
                    'target_ged': tensor([ 7.,  9.,  6.,  9., 13.,  9., ...]) # data[target_ged].shape为torch.Size([128])
                }
        
        e.g. 
            输入的data为{'g1': DataBatch(edge_index=[2, 10080], i=[560], x=[5600, 29], num_nodes=5600, batch=[5600], ptr=[561]), 'g2': DataBatch(edge_index=[2, 9898], i=[560], x=[4991, 29], num_nodes=4991, batch=[4991], ptr=[561]), 'target':xxx, 'target_ged':xxx
            # 整个batch的节点数为5600，边数为10080，i=[560],表示整个batch一共有560个图对 (i是图g的索引，形状是[1]； i=[560] 代表有560个图，即i的值与batch_size一致，已验证！)
            data["g1"].batch:tensor([  0,   0,   0,  ..., 559, 559, 559]), data["g2"].batch:tensor([  0,   0,   0,  ..., 559, 559, 559]),
            data["g1"].batch:tensor(
                 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,
                    1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,
                    2,   2,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   4,   4,
                    ...
                    555, 555, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 557, 557,
                    557, 557, 557, 557, 557, 557, 557, 557, 558, 558, 558, 558, 558, 558,
                    558, 558, 558, 558, 559, 559, 559, 559, 559, 559, 559, 559, 559, 559])
            data["g2"].batch:tensor([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,
                    1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,
                    3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   4,   4,   4,   4,
                    4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   6,   6,
                    ....
                    554, 554, 554, 555, 555, 555, 555, 555, 555, 555, 555, 555, 555, 556,
                    556, 556, 556, 556, 556, 556, 556, 557, 557, 557, 557, 557, 557, 557,
                    557, 557, 558, 558, 558, 558, 558, 558, 558, 558, 558, 559, 559, 559,
                    559, 559, 559, 559, 559, 559, 559])
            
            data["g1"].ptr:tensor([   0,   10,   20,   30,   40,   50,   60, .... 5560, 5570, 5580, 5590, 5600]) 
            data["g2"].ptr:tensor([   0,   10,   19,   28,   38,   46,   50, .... 4955, 4963, 4972, 4981, 4991])
            data["g1"].i:tensor([684, 684, 684, 684, 684, 684, 684, 684, 684, .... 684, 684, 684, 684, 684, 684])
            data["g2"].i:tensor([  0,   1,   2,   3,   4,   5,   6,   7, .... 554, 555, 556, 557, 558, 559])
            
        features_level1_1.shape: torch.Size([1133, 64])
        features_level1_2.shape: torch.Size([1121, 64])
        pooled_features_level1_1.shape: torch.Size([128, 64])
        pooled_features_level1_2.shape: torch.Size([128, 64])
        scores_level1.shape: torch.Size([128, 32])
        
        features_level2_1.shape: torch.Size([1133, 32])
        features_level2_2.shape: torch.Size([1121, 32])
        pooled_features_level2_1.shape: torch.Size([128, 32])
        pooled_features_level2_2.shape: torch.Size([128, 32])
        scores_level2.shape: torch.Size([128, 16])
        
        pooled_features_level3_1.shape: torch.Size([128, 16])
        pooled_features_level3_2.shape: torch.Size([128, 16])
        scores_level3.shape: torch.Size([128, 8])
        
        scores.shape: torch.Size([128, 56])
        最终返回的scores.shape=torch.Size([128, 16]) """
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
        # print('[EGSC-T/src/model.py] 分类器 self.scoring_layer(scores):', self.scoring_layer(score), 'scores:', score)
        # print(f'[EGSC-T/src/model.py] 分类器 EGSCT_classifier 正在执行forward函数 输入scores的维度为{scores.shape}, 输出score的维度为{score.shape}')
        #  输入scores的维度为torch.Size([128, 16]), 输出score的维度为torch.Size([128])
        return  score 