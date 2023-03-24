import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau
import time

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

# from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt
from thop import profile

from model_kd import EGSC_generator, EGSC_fusion, EGSC_fusion_classifier, EGSC_classifier, EGSC_teacher, logits_D, local_emb_D, global_emb_D

from model_kd_light import EGSC_generator as EGSC_generator_light
from model_kd_light import EGSC_fusion as EGSC_fusion_light
from model_kd_light import EGSC_fusion_classifier as EGSC_fusion_classifier_light
from model_kd_light import EGSC_classifier as EGSC_classifier_light
from model_kd_light import EGSC_teacher as EGSC_teacher_light
from model_kd_light import logits_D as logits_D_light
from model_kd_light import global_emb_D as global_emb_D_light


import pdb
from layers import RkdDistance, RKdAngle, repeat_certain_graph
import wandb
from torchviz import make_dot

class EGSC_KD_Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(self.args.cuda_id) if torch.cuda.is_available() else 'cpu')
        print('[EGSC_KD_Trainer] Using device:', self.device)
        self.best_rho = 0
        self.best_tau = 0
        self.best_prec_at_10 = 0
        self.best_prec_at_20 = 0
        self.best_model_error = float('inf')
        self.process_dataset()
        self.setup_model()


    def setup_model(self):
        if self.args.light:
            print('setup light model...')
            self.model_g = EGSC_generator_light(self.args, self.number_of_labels)
            self.model_f = EGSC_fusion_light(self.args, self.number_of_labels)
            self.model_c = EGSC_classifier_light(self.args, self.number_of_labels)
            self.model_c1 = EGSC_fusion_classifier_light(self.args, self.number_of_labels)
            self.model_g_fix = EGSC_teacher_light(self.args, self.number_of_labels) # number_of_labels = 29
            self.Discriminator = logits_D_light(self.args.adversarial_ouput_class, self.args.adversarial_ouput_class)
            # self.Discriminator_e = local_emb_D(n_hidden=16)
            self.Discriminator_g = global_emb_D_light(n_hidden=16)
        else:
            print('setup original model...')
            self.model_g = EGSC_generator(self.args, self.number_of_labels)
            self.model_f = EGSC_fusion(self.args, self.number_of_labels)
            self.model_c = EGSC_classifier(self.args, self.number_of_labels)
            self.model_c1 = EGSC_fusion_classifier(self.args, self.number_of_labels)
            self.model_g_fix = EGSC_teacher(self.args, self.number_of_labels) # number_of_labels = 29
            self.Discriminator = logits_D(self.args.adversarial_ouput_class, self.args.adversarial_ouput_class)
            # self.Discriminator_e = local_emb_D(n_hidden=16)
            self.Discriminator_g = global_emb_D(n_hidden=16)
        
        # if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        #         self.model_g = torch.nn.DataParallel(self.model_g) #多gpu训练,自动选择gpu device_ids=[0,1,2,3]
        #         self.model_f = torch.nn.DataParallel(self.model_f)
        #         self.model_c = torch.nn.DataParallel(self.model_c)
        #         self.model_c1 = torch.nn.DataParallel(self.model_c1)
        #         # self.model_g_fix = torch.nn.DataParallel(self.model_g_fix)
        #         self.Discriminator = torch.nn.DataParallel(self.Discriminator)
    
        self.get_parameter_number(self.model_g) # 【重要】查看学生模型的参数，total_num: 20995，trainable_num: 20995
        # self.get_parameter_number(self.model_f) # 模型参数，total_num: 52376，trainable_num: 52376
        # self.get_parameter_number(self.model_c) # 模型参数，total_num: 17，trainable_num: 17
        # self.get_parameter_number(self.model_c1) # 模型参数，total_num: 545，trainable_num: 545
        self.get_parameter_number(self.model_g_fix) # 【重要】查看教师模型的参数，total_num: 122038，trainable_num: 122038
        
        """ thop
        edge_index = torch.randint(1, 10, (2, 1600)) # https://runebook.dev/zh-CN/docs/pytorch/generated/torch.randint
        features = torch.randn(800, 29)
        batch = torch.randint(0, 799, (800,))
        macs, params = profile(self.model_g, inputs=(edge_index, features, batch))
        print('FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
        print('params: ', params)
        # 模型输入数据 edge_index_1.shape: torch.Size([2, 1600]) features_1.shape: torch.Size([800, 29]) batch_1.shape: torch.Size([800])
        # edge_index_1=tensor([[   0,    0,    1,  ..., 1150, 1150, 1150],
        #                     [   3,    8,    6,  ..., 1144, 1147, 1149]]), 
        # features_1=tensor([[0., 0., 1.,  ..., 0., 0., 0.],
        #                     [0., 0., 0.,  ..., 0., 0., 0.],
        #                     [1., 0., 0.,  ..., 0., 0., 0.],
        #                     ...,
        #                     [0., 0., 0.,  ..., 0., 0., 0.],
        #                     [0., 0., 0.,  ..., 0., 0., 0.],
        #                     [0., 0., 1.,  ..., 0., 0., 0.]]), 
        # batch_1=tensor([  0,   0,   0,  ..., 127, 127, 127])
         """
         
        self.model_g.to(self.device)
        self.model_f.to(self.device)
        self.model_c.to(self.device)
        self.model_c1.to(self.device)
        self.model_g_fix.to(self.device)
        self.Discriminator.to(self.device)
        # self.Discriminator_e.to(self.device)
        self.Discriminator_g.to(self.device)
        
        """ print(f'[EGSC-KD/src/egsc_kd.py] self.model_g: {self.model_g}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_f: {self.model_f}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_c: {self.model_c}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_c1: {self.model_c1}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_g_fix: {self.model_g_fix}') """
        
        """ --dataset AIDS700nef 打印结果
        
        [EGSC-KD/src/egsc_kd.py] self.model_g: EGSC_generator(
        (convolution_1): GINConv(nn=Sequential(
            (0): Linear(in_features=29, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_2): GINConv(nn=Sequential(
            (0): Linear(in_features=64, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_3): GINConv(nn=Sequential(
            (0): Linear(in_features=32, out_features=16, bias=True)
            (1): ReLU()
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (attention): AttentionModule()
        (attention_level2): AttentionModule()
        (attention_level1): AttentionModule()
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_f: EGSC_fusion(
        (score_attention): SEAttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=224, out_features=56, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=56, out_features=224, bias=True)
            (3): Sigmoid()
            )
        (feat_layer): Linear(in_features=224, out_features=112, bias=True)
        (fully_connected_first): Linear(in_features=112, out_features=16, bias=True)
        )
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_c: EGSC_classifier(
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_c1: EGSC_fusion_classifier(
        (feat_layer): Linear(in_features=32, out_features=16, bias=True)
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_g_fix: EGSC_teacher(
        (convolution_1): GINConv(nn=Sequential(
            (0): Linear(in_features=29, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_2): GINConv(nn=Sequential(
            (0): Linear(in_features=64, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_3): GINConv(nn=Sequential(
            (0): Linear(in_features=32, out_features=16, bias=True)
            (1): ReLU()
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (attention_level3): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=16, out_features=4, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=16, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=16, out_features=16, bias=True)
        )
        (attention_level2): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=32, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=32, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=32, out_features=32, bias=True)
        )
        (attention_level1): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=64, out_features=64, bias=True)
        )
        (cross_attention_level2): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=32, out_features=32, bias=False)
            (w_ks): Linear(in_features=32, out_features=32, bias=False)
            (w_vs): Linear(in_features=32, out_features=32, bias=False)
            (fc): Linear(in_features=32, out_features=32, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=32, out_features=32, bias=True)
            (w_2): Linear(in_features=32, out_features=32, bias=True)
            (layer_norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (cross_attention_level3): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=16, out_features=16, bias=False)
            (w_ks): Linear(in_features=16, out_features=16, bias=False)
            (w_vs): Linear(in_features=16, out_features=16, bias=False)
            (fc): Linear(in_features=16, out_features=16, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((16,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=16, out_features=16, bias=True)
            (w_2): Linear(in_features=16, out_features=16, bias=True)
            (layer_norm): LayerNorm((16,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (cross_attention_level4): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=8, out_features=8, bias=False)
            (w_ks): Linear(in_features=8, out_features=8, bias=False)
            (w_vs): Linear(in_features=8, out_features=8, bias=False)
            (fc): Linear(in_features=8, out_features=8, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((8,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=8, out_features=8, bias=True)
            (w_2): Linear(in_features=8, out_features=8, bias=True)
            (layer_norm): LayerNorm((8,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (tensor_network_level3): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=32, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=32, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=8, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (tensor_network_level2): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=16, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (tensor_network_level1): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=128, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=128, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=32, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (fully_connected_first): Linear(in_features=56, out_features=16, bias=True)
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        (score_attention): SEAttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=56, out_features=14, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=14, out_features=56, bias=True)
            (3): Sigmoid()
            )
        )
        ) """

        self.loss_RkdDistance = RkdDistance()
        self.loss_RKdAngle = RKdAngle()


    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'模型参数，total_num: {total_num}，trainable_num: {trainable_num}')

    def load_model(self):
        if self.args.checkpoint:
            PATH_g = self.args.checkpoint
        else:
            PATH_g = '../Checkpoints/G_EarlyFusion_Disentangle_' +str(self.args.dataset) +'_gin'+'_checkpoint.pth'
        
        print('Loading Teacher Model from {}'.format(PATH_g))
        
        # print('self.model_g_fix: ', self.model_g_fix, '\nteacher model:', torch.load(PATH_g, map_location=self.device))
        
        self.model_g_fix.load_state_dict(torch.load(PATH_g, map_location=self.device), strict=False) # load_state_dict使用 state_dict 反序列化模型参数字典。用来加载模型参数。将 state_dict 中的 parameters 和 buffers 复制到此 module 及其子节点中。 概况：给模型对象加载训练好的模型参数，即加载模型参数
        
        # self.model_g_fix.load_state_dict(torch.load(PATH_g), strict=False) # because use light model, so we need to load the model with strict=False
        
        print('Model Loaded')
        
    def process_dataset(self):
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.nged_matrix = self.training_graphs.norm_ged # nged_matrix是归一化的ged矩阵, 即每个元素都除以了最大的ged值。矩阵维度为 [num_graphs, num_graphs]，每个元素代表两个图之间的ged值。
        self.ged_matrix = self.training_graphs.ged # ged_matrix是ged矩阵，即没有归一化的ged矩阵，即每个元素都是原始的ged值，没有除以最大的ged值。

        self.real_data_size = self.nged_matrix.size(0) # real_data_size是数据集中的图的数量，即矩阵的行数，即矩阵的列数。
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_dataset函数，self.nged_matrix.shape = {self.nged_matrix.shape}, self.ged_matrix.shape = {self.ged_matrix.shape}, self.real_data_size = {self.real_data_size}')
        # AIDS700nef数据集打印结果：self.nged_matrix.shape = torch.Size([700, 700]), self.ged_matrix.shape = torch.Size([700, 700]), self.real_data_size = 700
        
        if self.args.synth:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)  
            
            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat((self.nged_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat((torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))
        
        if self.training_graphs[0].x is None: # 如果数据集中的图没有节点特征，则使用one-hot编码，即将每个节点的特征编码为一个one-hot向量，向量的维度为节点的度数。 self.training_graphs[0].x 是第一个图的节点特征，即第一个图的节点特征矩阵。x是一个二维矩阵，每一行代表一个节点的特征，每一列代表一个特征的维度。x的定义在torch_geometric.data.data.Data类中。
            max_degree = 0 # max_degree是数据集中图中节点的最大度数。
            for g in self.training_graphs + self.testing_graphs + (self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0: # g.edge_index.size(1)是图g中边的数量，即图g的边数。edge_index代表图g中边的索引，即图g中每条边的两个节点的索引。edge_index的定义在torch_geometric.data.data.Data类中。
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
        
        # labeling of synth data according to real data format     
        if self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g = one_hot_degree(g)
                g.i = g.i + real_data_size # .i是图g的索引，即图g在数据集中的索引。i的定义在torch_geometric.data.data.Data类中。
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size    
        self.number_of_labels = self.training_graphs.num_features #num_features 函数 Returns the number of features per node in the dataset  num_features是图中节点的特征维度，即图中节点的特征数。num_features的定义在torch_geometric.data.data.Data类中。
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_dataset函数，self.number_of_labels = {self.number_of_labels}') # AIDS700nef数据集打印结果：self.number_of_labels = 29
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_dataset函数，self.training_graphs[0] = {self.training_graphs[0]}')
        # AIDS数据集打印结果： self.training_graphs[0] = Data(edge_index=[2, 18], i=[1], x=[10, 29], num_nodes=10)
        # edge_index: 用于存储节点之间的边，形状是 [2, num_edges]（  使用稀疏的方式存储边关系（edge_index中边的存储方式，有两个list，第 1 个list是边的起始点，第 2 个list是边的目标节点））；
        # x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]；
        # y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]；
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_dataset函数，self.training_graphs = {self.training_graphs}, len(self.training_graphs) = {len(self.training_graphs)}')
        # AIDS数据集打印结果： self.training_graphs = AIDS700nef(560), len(self.training_graphs) = 560
        pass

    def create_batches(self):
        # print(f"self.args.synth = {self.args.synth}") # self.args.synth = False
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader)) # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象

    def transform(self, data):
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        
        normalized_ged = self.nged_matrix[ # nged_matrix是归一化的ged矩阵, 即每个元素都除以了最大的ged值。矩阵维度为 [num_graphs, num_graphs]，每个元素代表两个图之间的ged值。
            data[0]["i"].reshape(-1).tolist(), # reshape(-1)将矩阵转化为一维数组，tolist()将数组转化为list
            data[1]["i"].reshape(-1).tolist()
            ].tolist() 
        """ 解释：self.nged_matrix 是700*700的矩阵，对应整个数据集所有的图，每个元素代表两个图之间的ged值。
        data[0]["i"] 和 data[1]["i"] 分别是128个图的索引，即在700个图中的索引。
        data[0]["i"].reshape(-1).tolist() 和 data[1]["i"].reshape(-1).tolist() 分别是128个图的索引的list。
        self.nged_matrix[ data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist() ] 分别是128个图的ged值的list。
        tolist()将矩阵转化为list 
        
        举例：参考/home/zl525/.conda/envs/Efficient_Graph_Similarity_Computation/lib/python3.9/site-packages/torch_geometric/datasets/ged_dataset.py的print信息，
        self.ged[24][431]=14.0, self.ged[253][54]=7.0  即索引值为24的图和索引值为431的图的GED为14.0，索引值为253的图和索引值为54的图的GED为7
        再结合下方data[0]["i"].reshape(-1).tolist()列表的第一个元素为24，第二个元素为253，data[1]["i"].reshape(-1).tolist()列表的第一个元素为431，第二个元素为54，
        然后从new_data["target_ged"]的打印信息可以看出，第一个元素代表的是索引值为24的图和索引值为431的图的GED为14.0，第二个元素代表的是索引值为253的图和索引值为54的图的GED为7
        """
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float() #具体计算过程为：遍历normalized_ged中的每个元素，将其取负数，然后取e的指数，最后将其转化为tensor new_data["target"]代表的是两个图的相似度，即两个图的ged的指数负数 为什么要取指数负数呢？因为ged越小，两个图越相似，所以取指数负数，相似度越大，指数负数越小，即相似度越大，两个图越相似。
        
        ged = self.ged_matrix[ # ged_matrix是ged矩阵，即没有归一化的ged矩阵，即每个元素都是原始的ged值，没有除以最大的ged值。
            data[0]["i"].reshape(-1).tolist(),
            data[1]["i"].reshape(-1).tolist()
            ].tolist() 
        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float() # new_data["target_ged"]代表的是两个图的ged，即两个图的ged的值

        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行transform函数，data[0]["i"].reshape(-1).tolist() = {data[0]["i"].reshape(-1).tolist()}')
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行transform函数，data[1]["i"].reshape(-1).tolist() = {data[1]["i"].reshape(-1).tolist()}')
        # AIDS数据集打印结果
        """ data[0]["i"] = tensor([ 24, 253, 456, 517, 491, 316, 347, 119, 431,  76, 131, 295, 547, 145, ..., 118]) , # 一共128个元素，代表128个图的索引
        data[0]["i"].reshape(-1) = tensor([ 24, 253, 456, 517, 491, 316, 347, 119, 431,  76, 131, 295, 547, 145, ..., 118]) , 
        data[0]["i"].reshape(-1).tolist() = [24, 253, 456, 517, 491, 316, 347, 119, 431, 76, 131, 295, 547, 145, 393, 322, ..., 118] 
        data[1]["i"].reshape(-1).tolist() = [431, 54, 549, 421, 372, 283, 298, 213, 234, 264, 430, 171, 154, 219, 3, 492, 39, 250, 97, 167, 288, 225, 302, 77, 502, 90, 173, 146, 386, 19, 76, 500, 294, 101, 373, 156, 440, 403, 449, 124, 67, 73, 330, 259, 196, 119, 31, 240, 457, 270, 134, 227, 318, 159, 210, 26, 176, 192, 11, 0, 280, 220, 520, 490, 87, 75, 267, 517, 6, 290, 313, 496, 138, 390, 452, 115, 338, 25, 149, 305, 437, 148, 248, 337, 224, 470, 478, 503, 232, 195, 98, 536, 35, 481, 523, 126, 95, 510, 229, 221, 178, 197, 274, 246, 535, 205, 446, 180, 347, 150, 404, 43, 308, 120, 413, 530, 215, 475, 558, 551, 336, 544, 82, 359, 15, 100, 92, 131]
        """

        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行transform函数，normalized_ged= {normalized_ged}')
        # AIDS数据集打印结果
        # normalized_ged= [2.0, 0.9333333373069763, 1.0, 0.8888888955116272, 1.0526316165924072, 1.7647058963775635, 3.1666667461395264, 1.7333333492279053, 1.0, 1.0, 0.699999988079071, 1.0, 0.8235294222831726, 0.9473684430122375, 0.699999988079071, 1.2941176891326904, 1.384615421295166, 0.7058823704719543, 0.800000011920929, 1.0526316165924072, 0.5263158082962036, 1.1111111640930176, 0.8888888955116272, 0.800000011920929, 0.9411764740943909, 0.800000011920929, 0.9473684430122375, 1.1578947305679321, 0.9473684430122375, 1.600000023841858, 0.7777777910232544, 1.0, 1.1578947305679321, 1.0526316165924072, 0.7368420958518982, 0.8888888955116272, 2.0, 1.2222222089767456, 0.9473684430122375, 1.375, 1.625, 1.4117647409439087, 0.5, 1.125, 1.3333333730697632, 0.7368420958518982, 1.6470588445663452, 0.25, 0.800000011920929, 2.2857143878936768, 0.8421052694320679, 1.5555555820465088, 0.8999999761581421, 0.8999999761581421, 1.4117647409439087, 0.7368420958518982, 0.125, 0.9473684430122375, 0.8235294222831726, 1.1578947305679321, 0.800000011920929, 0.8999999761581421, 1.263157844543457, 1.1578947305679321, 1.0, 0.6666666865348816, 1.125, 0.6000000238418579, 0.8888888955116272, 0.9411764740943909, 1.0, 0.7777777910232544, 1.263157844543457, 1.7333333492279053, 1.0526316165924072, 1.2222222089767456, 1.2000000476837158, 1.5, 0.8421052694320679, 0.800000011920929, 1.1764706373214722, 1.1578947305679321, 2.8333332538604736, 0.6000000238418579, 0.30000001192092896, 0.7777777910232544, 1.263157844543457, 1.100000023841858, 1.0526316165924072, 0.9473684430122375, 1.076923131942749, 1.1111111640930176, 1.600000023841858, 1.0526316165924072, 1.0, 1.529411792755127, 2.0, 1.1111111640930176, 0.8421052694320679, 0.8421052694320679, 1.5, 0.800000011920929, 1.25, 1.0, 0.699999988079071, 1.0526316165924072, 0.875, 0.6666666865348816, 0.5, 0.8235294222831726, 1.058823585510254, 0.699999988079071, 0.9473684430122375, 0.6315789222717285, 0.6315789222717285, 0.8888888955116272, 0.5, 1.0526316165924072, 1.1578947305679321, 1.100000023841858, 0.9411764740943909, 1.5, 0.8888888955116272, 1.058823585510254, 1.1428571939468384, 0.9411764740943909, 1.1111111640930176, 0.7368420958518982]

        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行transform函数，data = {data}, new_data= {new_data}')
        # AIDS数据集打印结果
        # edge_index: 用于存储节点之间的边，形状是 [2, num_edges]（使用稀疏的方式存储边关系（edge_index中边的存储方式，有两个list，第 1 个list是边的起始点，第 2 个list是边的目标节点））；
        # i是图g的索引，形状是[1]； i=[128] 代表有128个图。因为batch_size=128，所以i=[128]；
        # x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]；
        # y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]；
        """ data = (
            DataBatch(edge_index=[2, 2284], i=[128], x=[1139, 29], num_nodes=1139, batch=[1139], ptr=[129]), 
            DataBatch(edge_index=[2, 2224], i=[128], x=[1128, 29], num_nodes=1128, batch=[1128], ptr=[129])), 
            
            new_data= {
                'g1': DataBatch(edge_index=[2, 2284], i=[128], x=[1139, 29], num_nodes=1139, batch=[1139], ptr=[129]), 
                'g2': DataBatch(edge_index=[2, 2224], i=[128], x=[1128, 29], num_nodes=1128, batch=[1128], ptr=[129]), 
                'target': tensor([0.1353, 0.3932, 0.3679, 0.4111, 0.3490, 0.1712, 0.0421, 0.1767, 0.3679,
                                    0.3679, 0.4966, 0.3679, 0.4389, 0.3878, 0.4966, 0.2741, 0.2504, 0.4937,
                                    0.4493, 0.3490, 0.5908, 0.3292, 0.4111, 0.4493, 0.3902, 0.4493, 0.3878,
                                    0.3141, 0.3878, 0.2019, 0.4594, 0.3679, 0.3141, 0.3490, 0.4786, 0.4111,
                                    0.1353, 0.2946, 0.3878, 0.2528, 0.1969, 0.2437, 0.6065, 0.3247, 0.2636,
                                    0.4786, 0.1926, 0.7788, 0.4493, 0.1017, 0.4308, 0.2111, 0.4066, 0.4066,
                                    0.2437, 0.4786, 0.8825, 0.3878, 0.4389, 0.3141, 0.4493, 0.4066, 0.2828,
                                    0.3141, 0.3679, 0.5134, 0.3247, 0.5488, 0.4111, 0.3902, 0.3679, 0.4594,
                                    0.2828, 0.1767, 0.3490, 0.2946, 0.3012, 0.2231, 0.4308, 0.4493, 0.3084,
                                    0.3141, 0.0588, 0.5488, 0.7408, 0.4594, 0.2828, 0.3329, 0.3490, 0.3878,
                                    0.3406, 0.3292, 0.2019, 0.3490, 0.3679, 0.2167, 0.1353, 0.3292, 0.4308,
                                    0.4308, 0.2231, 0.4493, 0.2865, 0.3679, 0.4966, 0.3490, 0.4169, 0.5134,
                                    0.6065, 0.4389, 0.3469, 0.4966, 0.3878, 0.5318, 0.5318, 0.4111, 0.6065,
                                    0.3490, 0.3141, 0.3329, 0.3902, 0.2231, 0.4111, 0.3469, 0.3189, 0.3902,
                                    0.3292, 0.4786]), 
                'target_ged': tensor([14.,  7.,  9.,  8., 10., 15., 19., 13.,  9.,  9.,  7.,  8.,  7.,  9.,
                                        7., 11.,  9.,  6.,  8., 10.,  5., 10.,  8.,  6.,  8.,  8.,  9., 11.,
                                        9., 12.,  7.,  9., 11., 10.,  7.,  8., 14., 11.,  9., 11., 13., 12.,
                                        5.,  9., 12.,  7., 14.,  1.,  8., 16.,  8., 14.,  9.,  9., 12.,  7.,
                                        1.,  9.,  7., 11.,  8.,  9., 12., 11.,  9.,  6.,  9.,  6.,  8.,  8.,
                                        7.,  7., 12., 13., 10., 11., 12., 12.,  8.,  6., 10., 11., 17.,  6.,
                                        3.,  7., 12., 11., 10.,  9.,  7., 10., 12., 10., 10., 13., 16., 10.,
                                        8.,  8., 12.,  8., 10., 10.,  7., 10.,  7.,  6.,  5.,  7.,  9.,  7.,
                                        9.,  6.,  6.,  8.,  4., 10., 11., 11.,  8., 12.,  8.,  9.,  8.,  8.,
                                        10.,  7.]
                                     )
                } """

        return new_data

    def process_batch(self, data, epoch):
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_c1.zero_grad()
        
        data = self.transform(data)
        target = data["target"].to(self.device)

        edge_index_1 = data["g1"].edge_index.to(self.device)
        edge_index_2 = data["g2"].edge_index.to(self.device)
        features_1 = data["g1"].x.to(self.device)
        features_2 = data["g2"].x.to(self.device)
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        batch_1 = batch_1.to(self.device)
        batch_2 = batch_2.to(self.device)
        i_1 = data["g1"].i.to(self.device)
        i_2 = data["g2"].i.to(self.device)
        
        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

        prediction = self.model_c(self.model_f(pooled_features_1_all, pooled_features_2_all)).to(self.device)
        loss_reg = F.mse_loss(prediction, target, reduction='sum') #* 0.5

        pooled_features_1 = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
        pooled_features_2 = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

        feat_joint = self.model_f(pooled_features_1, pooled_features_2)
        feat_joint_1 = self.model_f(pooled_features_1, pooled_features_1)
        feat_joint_2 = self.model_f(pooled_features_2, pooled_features_2)

        feat_joint_fix = self.model_g_fix(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2).detach()
        feat_joint_fix_1 = self.model_g_fix(edge_index_1, features_1, batch_1, i_1, edge_index_1, features_1, batch_1, i_1).detach()
        feat_joint_fix_2 = self.model_g_fix(edge_index_2, features_2, batch_2, i_2, edge_index_2, features_2, batch_2, i_2).detach()
        feat_1 = feat_joint-feat_joint_1
        feat_fix_1 = feat_joint_fix-feat_joint_fix_1
        feat_2 = feat_joint-feat_joint_2
        feat_fix_2 = feat_joint_fix-feat_joint_fix_2

        if self.args.mode == "l1":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 10

        elif self.args.mode == "rkd_dis":
            loss_kd = (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 10
        elif self.args.mode == "rkd_ang":
            loss_kd = (self.loss_RKdAngle(feat_1, feat_fix_1) + \
            self.loss_RKdAngle(feat_2, feat_fix_2)) * 10

        elif self.args.mode == "both":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 5 + \
            (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 5

        else:
            loss_kd = 0

        feat_12 = torch.cat((feat_1, feat_2),dim=1)
        prediction_rec = self.model_c1(feat_12)

        loss_reg_rec = F.mse_loss(prediction, target, reduction='sum') # Apart from the KD loss, there is a supervision (i.e., MSE) loss LSreg on the student model to fulfill the object of the task.

        loss = loss_reg + loss_kd + loss_reg_rec # loss_reg是loss_regression loss_kd是loss_knowledge_distillation
        if self.args.wandb:
            wandb.config.stu_loss_type = "loss_reg+loss_kd+loss_reg_rec"
            wandb.log({
                "loss_reg": loss_reg.item(),
                "loss_kd": loss_kd.item(),
                "loss_reg_rec": loss_reg_rec.item(),
                "epoch": epoch,
            })

        loss.backward()
        self.optimizer_c1.step()
        self.optimizer_c.step()
        self.optimizer_f.step()
        self.optimizer_g.step()
        return loss_reg.item(), loss_kd.item()
    
    def process_batch_adversarial(self, data, epoch):
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_c1.zero_grad()
        self.opt_D.zero_grad()
        
        data = self.transform(data)
        target = data["target"].to(self.device) # target计算原理: 遍历normalized_ged中的每个元素，将其取负数，然后取e的指数，最后将其转化为tensor, 因此target的值域为[0,1]，ged越大，target越小, 即target越小相似度越小，target越大相似度越大。normalized_ged的定义：/home/zl525/.conda/envs/Efficient_Graph_Similarity_Computation/lib/python3.9/site-packages/torch_geometric/datasets/ged_dataset.py：self.norm_ged = torch.load(path) path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')

        edge_index_1 = data["g1"].edge_index.to(self.device) # edge_index_1是g1的边索引，edge_index_2是g2的边索引 edge_index=[2, xxx]，其中第1个list是边的起点，第二个list是边的终点
        edge_index_2 = data["g2"].edge_index.to(self.device)
        features_1 = data["g1"].x.to(self.device)  # x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]  AIDS数据集：features_1.shape: torch.Size([800, 29]) 800 nodes, 29 features
        features_2 = data["g2"].x.to(self.device)
        test1 = hasattr(data["g1"], 'batch') # test1: True
        test2 = hasattr(data["g2"], 'batch') # test2: True
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        batch_1 = batch_1.to(self.device)
        batch_2 = batch_2.to(self.device)
        i_1 = data["g1"].i.to(self.device)
        i_2 = data["g2"].i.to(self.device)

        # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，输入model_g之前 edge_index_1.shape={edge_index_1.shape}, features_1.shape={features_1.shape}, batch_1.shape={batch_1.shape}, edge_index_2.shape={edge_index_2.shape}, features_2.shape={features_2.shape}, batch_2.shape={batch_2.shape}")
        # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，输入model_g之前 edge_index_1={edge_index_1}, features_1={features_1}, batch_1={batch_1}")
        
        

        # # 统计计算当前循环的batch时，模型参数量和计算量 https://www.cnblogs.com/picassooo/p/16343737.html
        # macs, params = profile(self.model_g, inputs=(edge_index_1, features_1, batch_1, i_1))
        # print('[thop-profile] model_g FLOPs-1: ', macs*2)   # 一般来讲，FLOPs是macs的两倍 
        # print('[thop-profile] model_g params-1: ', params) 
        # macs, params = profile(self.model_g, inputs=(edge_index_2, features_2, batch_2, i_2))
        # print('[thop-profile] model_g FLOPs-2: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
        # print('[thop-profile] model_g params-2: ', params) 
        # macs, params = profile(self.model_g_fix, inputs=(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2))
        # print('[thop-profile] model_g_fix FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
        # print('[thop-profile] model_g_fix params: ', params) 
        """ 某一个batch的输出结果如下：
        [thop-profile] model_g FLOPs-1:  8581120.0
        [thop-profile] model_g params-1:  10240.0
        [thop-profile] model_g FLOPs-2:  8765440.0
        [thop-profile] model_g params-2:  10240.0
        [thop-profile] model_g_fix FLOPs:  25749248.0
        [thop-profile] model_g_fix params:  53810.0
        """
        
        
        """ 输入model_g之前的数据溯源和流转过程如下：
        
        create_batches函数如下：
            def create_batches(self):
                
                source_loader = DataLoader(self.training_graphs.shuffle() + [], batch_size=self.args.batch_size) # batch_size=128
                target_loader = DataLoader(self.training_graphs.shuffle() + [], batch_size=self.args.batch_size)
                
                return list(zip(source_loader, target_loader))
        
        batches = self.create_batches() # batches是一个list，每个元素是一个tuple，tuple中有两个元素，每个元素是一个DataBatch对象，DataBatch对象中有edge_index, x, batch, ptr, i等属性
        
        for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
            if self.args.use_adversarial == 1:
                loss_score, loss_score_kd = self.process_batch_adversarial(batch_pair, epoch)
        
        data = batch_pair
        
        data["g1"] = data[0]
        data["g2"] = data[1]
        data["g1"]: DataBatch(edge_index=[2, 2284], i=[128], x=[1139, 29], num_nodes=1139, batch=[1139], ptr=[129]), 
        data["g2"]: DataBatch(edge_index=[2, 2224], i=[128], x=[1128, 29], num_nodes=1128, batch=[1128], ptr=[129]), 
        
        edge_index_1 = data["g1"].edge_index # 边索引
        features_1 = data["g1"].x  # x表示节点特征
        batch_1 = data["g1"].batch # batch表示节点个数
        
        edge_index_1.shape=torch.Size([2, 2276]), features_1.shape=torch.Size([1151, 29]), batch_1.shape=torch.Size([1151]),  # 1151代表1151个节点，29代表29个特征，2276代表2276条边
        edge_index_2.shape=torch.Size([2, 2256]), features_2.shape=torch.Size([1133, 29]), batch_2.shape=torch.Size([1133])
        
        edge_index_1=tensor([[   0,    0,    1,  ..., 1150, 1150, 1150],
                            [   3,    8,    6,  ..., 1144, 1147, 1149]]), 
        features_1=tensor([[0., 0., 1.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [1., 0., 0.,  ..., 0., 0., 0.],
                            ...,
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 1.,  ..., 0., 0., 0.]]), 
        batch_1=tensor([  0,   0,   0,  ..., 127, 127, 127])
        """
        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

        # 使用torchviz 查看pytorc计算图 可视化
        # model_structure = make_dot(pooled_features_1_all) # 生成模型网络结构图, 可选参数params=dict(model.named_parameters())，用于显示模型的参数名和参数值。
        # model_structure.render('torch_dot_graph1', view=False, format='png') # 生成模型网络结构图，保存为model_structure.png文件，view=True表示显示模型的计算图。

        prediction = self.model_c(self.model_f(pooled_features_1_all, pooled_features_2_all)).to(self.device)
        loss_reg = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，prediction.shape={prediction.shape}, target.shape={target.shape}') # AIDS数据集：prediction.shape=torch.Size([128]), target.shape=torch.Size([128])
        # label_loss = self.loss_fcn(prediction, target)
        label_loss = loss_reg

        pooled_features_1 = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
        pooled_features_2 = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

        feat_joint = self.model_f(pooled_features_1, pooled_features_2)
        feat_joint_1 = self.model_f(pooled_features_1, pooled_features_1)
        feat_joint_2 = self.model_f(pooled_features_2, pooled_features_2)
        feat_1 = feat_joint-feat_joint_1
        feat_2 = feat_joint-feat_joint_2
        
        # 测试
        test = self.model_c(feat_joint)
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，feat_joint.shape={feat_joint.shape}, feat_joint_1.shape={feat_joint_1.shape}, feat_joint_2.shape={feat_joint_2.shape}, test.shape={test.shape}')
        # feat_joint.shape=torch.Size([128, 16]), feat_joint_1.shape=torch.Size([128, 16]), feat_joint_2.shape=torch.Size([128, 16]), test.shape=torch.Size([128])
        # 128是batch_size，16是model_f的输出维度

        # model_g_fix 相当于完成了学生网络模型里的model_g和model_f的融合
        feat_joint_fix = self.model_g_fix(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2).detach()
        feat_joint_fix_1 = self.model_g_fix(edge_index_1, features_1, batch_1, i_1, edge_index_1, features_1, batch_1, i_1).detach()
        feat_joint_fix_2 = self.model_g_fix(edge_index_2, features_2, batch_2, i_2, edge_index_2, features_2, batch_2, i_2).detach()
        feat_fix_1 = feat_joint_fix-feat_joint_fix_1
        feat_fix_2 = feat_joint_fix-feat_joint_fix_2

        if self.args.mode == "l1":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 10

        elif self.args.mode == "rkd_dis":
            loss_kd = (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 10
        elif self.args.mode == "rkd_ang":
            loss_kd = (self.loss_RKdAngle(feat_1, feat_fix_1) + \
            self.loss_RKdAngle(feat_2, feat_fix_2)) * 10

        elif self.args.mode == "both":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 5 + \
            (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 5

        else:
            loss_kd = 0

        feat_12 = torch.cat((feat_1, feat_2),dim=1)
        prediction_rec = self.model_c1(feat_12)

        loss_reg_rec = F.mse_loss(prediction, target, reduction='sum') # Apart from the KD loss, there is a supervision (i.e., MSE) loss LSreg on the student model to fulfill the object of the task.

        # ============================================
        #  Train Dis 即训练判别器，用来判断生成器生成的样本是真样本还是假样本。对这些生成的图片，G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此这样的训练就叫做 对抗训练(adversarial training), 也叫做GAN。 https://zhuanlan.zhihu.com/p/114838349
        # ============================================
        if epoch % self.args.d_critic == 0:
            loss_D = 0
            ## distinguish by Dl 
            # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，正在训练判别器 feat_1.shape={feat_1.shape}, feat_fix_1.shape={feat_fix_1.shape}, feat_2.shape={feat_2.shape}, feat_fix_2.shape={feat_fix_2.shape}, feat_1={feat_1}, feat_fix_1={feat_fix_1}, feat_2={feat_2}, feat_fix_2={feat_fix_2}")
            """ AIDS数据集打印结果：
            feat_1.shape=torch.Size([128, 16]), feat_fix_1.shape=torch.Size([128, 16]), feat_2.shape=torch.Size([128, 16]), feat_fix_2.shape=torch.Size([128, 16])
            feat_1=tensor([[ 0.6670, -0.1259, -0.1003,  ...,  0.0000,  0.0000, -0.2921],
                            [-0.1005,  0.5296,  0.2563,  ...,  0.0547,  0.0000,  0.2141],
                            [ 0.0000,  0.6043, -0.0814,  ...,  0.4275,  0.0000, -0.1944],
                            ...,
                            [ 0.0000, -0.6059, -0.1190,  ...,  0.0000,  0.0000,  0.2649],
                            [ 0.0000, -1.8759,  0.1510,  ...,  0.0000,  0.0000,  0.0000],
                            [-0.0695, -0.9954, -0.1967,  ...,  0.0000,  0.0000,  0.0000]],
                            grad_fn=<SubBackward0>), 
            feat_fix_1=tensor([[-1.3189,  1.3408,  0.0000,  ...,  1.7653,  0.0000, -0.3063],
                                [-1.6644, -0.2901,  0.0000,  ...,  0.0000,  0.0000, -0.3853],
                                [-1.7918, -0.1946,  0.0000,  ...,  0.3683,  0.0000, -0.5966],
                                ...,
                                [-1.8001, -0.5275,  0.0000,  ...,  0.0000,  0.0000, -0.6037],
                                [-2.1630, -0.3862,  0.0000,  ...,  0.8493,  0.0000, -0.9134],
                                [-2.3048, -0.2304,  0.0000,  ...,  0.0000,  0.0000, -0.6950]]), 
            feat_2=tensor([[-0.6197, -0.0446,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                            [ 0.0000,  1.0848,  0.4241,  ..., -0.0880,  0.0000,  0.8468],
                            [ 0.0000,  0.2137, -0.0743,  ..., -0.2114,  0.0000,  0.0000],
                            ...,
                            [ 0.0000, -0.2840,  0.0838,  ...,  0.0000,  0.0000,  0.3631],
                            [ 0.0000,  1.1574,  1.4006,  ...,  0.0000,  0.0000,  0.0000],
                            [ 0.0000,  0.5326, -0.4758,  ...,  0.0000,  0.0000, -1.0596]],
                        grad_fn=<SubBackward0>), 
            feat_fix_2=tensor([[-2.4916, -0.7001,  0.0000,  ...,  1.7653,  0.0000, -0.4123],
                                [-1.8187, -0.2487,  0.0000,  ...,  0.0000,  0.0000, -0.6387],
                                [-2.0963, -0.4135,  0.0000,  ...,  0.3683,  0.0000, -0.5459],
                                ...,
                                [-2.0352, -0.2346,  0.0000,  ...,  0.0000,  0.0000, -0.6191],
                                [-1.9456, -0.6408,  0.0000,  ...,  0.8493,  0.0000, -0.5140],
                                [-2.4196, -0.3791,  0.0000,  ...,  0.0000,  0.0000, -0.8565]]) """
            
            if self.args.adversarial_ouput_class == 1:
                ###################### distinguish by Dl
                # test option1: use prediction as stu_logits and target as tea_logits
                stu_logits = prediction.unsqueeze(1)
                tea_logits = target.unsqueeze(1)
                # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数 stu_logits.shape={stu_logits.shape}, tea_logits.shape={tea_logits.shape} stu_logits={stu_logits}, tea_logits={tea_logits}')
                self.Discriminator.train() # self.Discriminator本质是一个logits判别器，用来判断生成器生成的样本是真样本还是假样本。
                stu_logits = stu_logits.detach() # logits.detach()是将logits的梯度设置为False，即不计算logits的梯度。为什么不计算logits的梯度呢？因为logits是生成器生成的样本，而生成器的目的是生成样本，而不是生成样本的梯度。所以不计算logits的梯度。？TODO
                pos_z = self.Discriminator(tea_logits) # 将教师模型的输出tea_logits输入到logits判别器中，得到pos_z。这里的pos_z是一个二维张量，表示真样本的logits判别器的输出。
                neg_z = self.Discriminator(stu_logits) # 将学生模型的输出stu_logits输入到logits判别器中，得到neg_z。这里的neg_z是一个二维张量，表示假样本的logits判别器的输出。
                real_z = torch.sigmoid(pos_z[:, -1]) # 这里的real_z是一个一维张量，表示真样本的logits判别器的输出的最后一列，即真样本的logits判别器的输出的最后一个元素。[:, -1] it means you are taking all the rows and only the last column. -1 represents the last column. 
                fake_z = torch.sigmoid(neg_z[:, -1]) # torch.sigmoid()是一个Sigmoid函数，用来将输入的张量中的每个元素转换为0到1之间的数。假样本的logits判别器的输出的最后一个元素。
                # 关键点：能将real_z和全1矩阵进行loss计算的原因是real_z这个张量的每个元素都是pos_z的最后一个元素，代表了real/fake的判别结果，即对于判别器来说判断误差就是它认为的real的概率和1的差距，fake的概率和0的差距
                ad_loss = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) # 对抗损失计算逻辑：真样本的logits判别器的输出的最后一个元素与1的交叉熵损失 + 假样本的logits判别器的输出的最后一个元素与0的交叉熵损失。为什么这样算呢？因为真样本的logits判别器的输出的最后一个元素应该趋近于1，而假样本的logits判别器的输出的最后一个元素应该趋近于0。
                """ # 验证测试，将real_z和全0矩阵进行loss计算，将fake_z和全1矩阵进行loss计算，预计会影响最后的准确率？
                ad_loss_1 = self.loss_dis(real_z, torch.zeros_like(real_z)) + self.loss_dis(fake_z, torch.ones_like(fake_z))
                wandb.config.note="验证测试 dicriminator test" """
                
                ###################### distinguish by De (Discriminator_embedding)
                # # graph1
                # tea_model_node_emb = feat_fix_1.detach() # 教师模型的节点embedding
                # stu_model_node_emb = feat_1.detach() # 学生模型的节点embedding feat_joint.shape=torch.Size([128, 16])
                # self.Discriminator_e.train()
                # print(f"tea_model_node_emb.shape={tea_model_node_emb.shape}, stu_model_node_emb.shape={stu_model_node_emb.shape}, data['g1']={data['g1']}")
                # # AIDS数据集打印结果: tea_model_node_emb.shape=torch.Size([128, 16]), stu_model_node_emb.shape=torch.Size([128, 16]), data['g1']=DataBatch(edge_index=[2, 2276], i=[128], x=[1151, 29], num_nodes=1151, batch=[1151], ptr=[129])
                # pos_e = self.Discriminator_e(tea_model_node_emb, data["g1"])
                # neg_e = self.Discriminator_e(stu_model_node_emb, data["g1"])
                # real_e = torch.sigmoid(pos_e)
                # fake_e = torch.sigmoid(neg_e)
                # ad_eloss_1 = self.loss_dis(real_e, torch.ones_like(real_e)) + self.loss_dis(fake_e, torch.zeros_like(fake_e))
                
                ###################### distinguish by Dg (Discriminator_graph)
                # tea_model_node_emb = feat_fix_1.detach()
                # stu_model_node_emb = feat_1.detach() 

                tea_model_node_emb = feat_joint_fix.detach()
                stu_model_node_emb = feat_joint.detach()
                
                self.Discriminator_g.train()
                tea_sum = torch.sigmoid(tea_model_node_emb.mean(dim=0)).unsqueeze(-1) # tea_sum.shape=torch.Size([16, 1]) unsqueeze(-1)指在最后一维增加一个维度 mean(dim=0)指在第0维求均值
                pos_g = self.Discriminator_g(tea_model_node_emb, tea_sum)
                neg_g = self.Discriminator_g(stu_model_node_emb, tea_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = self.loss_dis(real_g, torch.ones_like(real_g)) + self.loss_dis(fake_g, torch.zeros_like(fake_g))
                # print('ad_gloss1:', ad_gloss1) # tensor(1.2143, grad_fn=<AddBackward0>)
                
                self.Discriminator_g.train()
                stu_sum = torch.sigmoid(stu_model_node_emb.mean(dim=0)).unsqueeze(-1) # stu_sum.shape=torch.Size([16, 1])
                pos_g = self.Discriminator_g(tea_model_node_emb, stu_sum)
                neg_g = self.Discriminator_g(stu_model_node_emb, stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = self.loss_dis(real_g, torch.ones_like(real_g)) + self.loss_dis(fake_g, torch.zeros_like(fake_g))
                
                
                loss_D = ad_loss + ad_gloss1 + ad_gloss2
            elif self.args.adversarial_ouput_class == 16:
                # test option2: use feat_1 as stu_logits and feat_fix_1 as tea_logits
                stu_logits_1 = feat_1
                tea_logits_1 = feat_fix_1
                stu_logits_2 = feat_2
                tea_logits_2 = feat_fix_2
            
                self.Discriminator.train() # self.Discriminator本质是一个logits判别器，用来判断生成器生成的样本是真样本还是假样本。
                stu_logits_1 = stu_logits_1.detach() # logits.detach()是将logits的梯度设置为False，即不计算logits的梯度。为什么不计算logits的梯度呢？因为logits是生成器生成的样本，而生成器的目的是生成样本，而不是生成样本的梯度。所以不计算logits的梯度。？TODO
                pos_z = self.Discriminator(tea_logits_1) # 将教师模型的输出tea_logits输入到logits判别器中，得到pos_z。这里的pos_z是一个二维张量，表示真样本的logits判别器的输出。
                neg_z = self.Discriminator(stu_logits_1) # 将学生模型的输出stu_logits输入到logits判别器中，得到neg_z。这里的neg_z是一个二维张量，表示假样本的logits判别器的输出。
                real_z = torch.sigmoid(pos_z[:, -1]) # 这里的real_z是一个一维张量，表示真样本的logits判别器的输出的最后一列，即真样本的logits判别器的输出的最后一个元素。[:, -1] it means you are taking all the rows and only the last column. -1 represents the last column. 
                fake_z = torch.sigmoid(neg_z[:, -1]) # torch.sigmoid()是一个Sigmoid函数，用来将输入的张量中的每个元素转换为0到1之间的数。假样本的logits判别器的输出的最后一个元素。
                # 关键点：能将real_z和全1矩阵进行loss计算的原因是real_z这个张量的每个元素都是pos_z的最后一个元素，代表了real/fake的判别结果，即对于判别器来说判断误差就是它认为的real的概率和1的差距，fake的概率和0的差距
                ad_loss_1 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) # 对抗损失计算逻辑：真样本的logits判别器的输出的最后一个元素与1的交叉熵损失 + 假样本的logits判别器的输出的最后一个元素与0的交叉熵损失。为什么这样算呢？因为真样本的logits判别器的输出的最后一个元素应该趋近于1，而假样本的logits判别器的输出的最后一个元素应该趋近于0。
                """ # 验证测试，将real_z和全0矩阵进行loss计算，将fake_z和全1矩阵进行loss计算，预计会影响最后的准确率？
                ad_loss_1 = self.loss_dis(real_z, torch.zeros_like(real_z)) + self.loss_dis(fake_z, torch.ones_like(fake_z))
                wandb.config.note="验证测试 dicriminator test" """
                
                stu_logits_2 = stu_logits_2.detach() # logits.detach()是将logits的梯度设置为False，即不计算logits的梯度。为什么不计算logits的梯度呢？因为logits是生成器生成的样本，而生成器的目的是生成样本，而不是生成样本的梯度。所以不计算logits的梯度。？TODO
                pos_z = self.Discriminator(tea_logits_2) # 将教师模型的输出tea_logits输入到logits判别器中，得到pos_z。这里的pos_z是一个二维张量，表示真样本的logits判别器的输出。
                neg_z = self.Discriminator(stu_logits_2) # 将学生模型的输出stu_logits输入到logits判别器中，得到neg_z。这里的neg_z是一个二维张量，表示假样本的logits判别器的输出。
                real_z = torch.sigmoid(pos_z[:, -1]) # 这里的real_z是一个一维张量，表示真样本的logits判别器的输出的最后一列，即真样本的logits判别器的输出的最后一个元素。[:, -1] it means you are taking all the rows and only the last column. -1 represents the last column. 
                fake_z = torch.sigmoid(neg_z[:, -1]) # torch.sigmoid()是一个Sigmoid函数，用来将输入的张量中的每个元素转换为0到1之间的数。假样本的logits判别器的输出的最后一个元素。
                ad_loss_2 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) # 对抗损失计算逻辑：真样本的logits判别器的输出的最后一个元素与1的交叉熵损失 + 假样本的logits判别器的输出的最后一个元素与0的交叉熵损失。为什么这样算呢？因为真样本的logits判别器的输出的最后一个元素应该趋近于1，而假样本的logits判别器的输出的最后一个元素应该趋近于0。
                # 验证测试，将real_z和全0矩阵进行loss计算，将fake_z和全1矩阵进行loss计算，预计会影响最后的准确率？
                # ad_loss_2 = self.loss_dis(real_z, torch.zeros_like(real_z)) + self.loss_dis(fake_z, torch.ones_like(fake_z))
                
                loss_D = 0.5 * (ad_loss_1 + ad_loss_2) # ad_loss全称是adversarial loss，即对抗损失，ds_loss全称是discriminator loss，即判别器损失。loss_D是对抗损失和判别器损失的加权和。
            else:
                raise ValueError("error: adversarial_ouput_class should be 1 or 16")

            # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数， 正在训练判别器 pos_z.shape={pos_z.shape} neg_z.shape={neg_z.shape} pos_z[:, -1].shape = {pos_z[:, -1].shape} neg_z[:, -1].shape = {neg_z[:, -1].shape}, real_z.shape={real_z.shape}, fake_z.shape={fake_z.shape}")
            # pos_z.shape=torch.Size([128, 17]) neg_z.shape=torch.Size([128, 17]) pos_z[:, -1].shape = torch.Size([128]) neg_z[:, -1].shape = torch.Size([128]), real_z.shape=torch.Size([128]), fake_z.shape=torch.Size([128])
            
            # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数， 正在训练判别器 pos_z={pos_z} neg_z={neg_z} pos_z[:, -1] = {pos_z[:, -1]} neg_z[:, -1] = {neg_z[:, -1]}, real_z={real_z}, fake_z={fake_z}")
            """ pos_z=tensor([[-0.4931, -1.4453,  0.8155,  ..., -2.5893,  2.6601, -0.5236],
                                [ 0.1476, -0.3344,  0.1188,  ..., -0.3710,  0.5252, -0.4145],
                                [ 0.1326, -0.5620,  0.2250,  ..., -0.9009,  1.0130, -0.5323],
                                ...,
                                [ 0.1397, -0.4185,  0.0142,  ..., -0.4749,  0.3507, -0.4349],
                                [ 0.0668, -0.9032,  0.2863,  ..., -1.2639,  1.0375, -0.4141],
                                [ 0.0825, -0.3074,  0.0518,  ..., -0.2603,  0.4037, -0.4159]],
                                grad_fn=<MmBackward0>) 
                
                neg_z=tensor([[-0.0841,  0.1354,  0.0215,  ...,  0.0412,  0.1980, -0.0833],
                                [-0.3040, -0.1549,  0.1010,  ...,  0.0668,  0.0131, -0.1552],
                                [-0.2693, -0.1267, -0.1673,  ..., -0.0054, -0.0558, -0.2367],
                                ...,
                                [-0.1033,  0.0292,  0.0238,  ...,  0.0042, -0.0290, -0.1174],
                                [ 0.0055, -0.0704,  0.1827,  ...,  0.0915,  0.2511, -0.0928],
                                [-0.1461,  0.1189,  0.0167,  ...,  0.1492,  0.0117, -0.0844]],
                                grad_fn=<MmBackward0>) 
                
                pos_z[:, -1] = tensor([-0.5236, -0.4145, -0.5323, -0.5141, -0.4634, -0.8289, -1.4474, -0.6970,
                                        -0.5446, -0.4084, -0.4145, -0.5489, -0.3823, -0.5045, -0.4458, -0.4816,
                                        -0.5233, -0.3913, -0.3428, -0.3925, -0.2856, -0.4965, -0.4479, -0.3498,
                                        -0.4499, -0.3567, -0.5485, -0.5783, -0.4292, -0.6643, -0.4681, -0.5811,
                                        -0.5118, -0.5228, -0.3375, -0.5199, -1.2706, -0.4480, -0.4136, -0.5500,
                                        -0.8860, -0.5053, -0.1975, -0.5023, -0.4879, -0.2582, -0.5496, -0.3627,
                                        -0.2998, -0.6414, -0.3670, -0.6816, -0.4655, -0.4315, -0.8074, -0.3927,
                                        -0.1433, -0.4036, -0.3299, -0.6001, -0.3572, -0.3709, -0.5042, -0.4905,
                                        -0.4838, -0.3514, -0.5271, -0.4233, -0.4679, -0.5608, -0.4839, -0.4845,
                                        -0.5257, -0.5046, -0.4374, -0.5308, -0.5023, -0.8156, -0.4210, -0.3319,
                                        -0.4377, -0.5330, -1.1385, -0.2994, -0.1840, -0.4673, -0.5014, -0.5060,
                                        -0.4964, -0.4274, -0.4696, -0.5764, -0.4613, -0.4882, -0.5154, -0.5419,
                                        -1.0300, -0.5393, -0.4378, -0.4252, -0.6008, -0.4325, -0.5793, -0.6083,
                                        -0.4581, -0.3821, -0.5021, -0.3868, -0.1985, -0.3499, -0.6001, -0.3474,
                                        -0.4355, -0.3430, -0.2669, -0.4256, -0.2510, -0.4666, -0.5498, -0.5245,
                                        -0.5089, -0.6002, -0.4942, -0.4622, -0.5606, -0.4349, -0.4141, -0.4159],
                                        grad_fn=<SelectBackward0>) 
                
                neg_z[:, -1] = tensor([-0.0833, -0.1552, -0.2367, -0.1055, -0.1208, -0.0194, -0.0788, -0.2829,
                                        -0.3946, -0.5004, -0.1603, -0.2712, -0.1873, -0.2818, -0.2309,  0.0016,
                                        -0.2275,  0.0728, -0.2872, -0.5654, -0.1249, -0.3139, -0.1970, -0.1579,
                                        -0.2107, -0.0782, -0.2399, -0.2070, -0.1506, -0.2329, -0.1599,  0.0199,
                                        -0.1717, -0.1339, -0.0400, -0.2688, -0.2120, -0.2773, -0.1281, -0.5037,
                                        -0.1123, -0.2193, -0.1331, -0.1077, -0.0054, -0.0490, -0.2720, -0.0430,
                                        -0.3129, -0.5721, -0.1590, -0.3346, -0.2096, -0.3793, -0.2156, -0.1441,
                                        -0.1431, -0.5423, -0.3565, -0.5837, -0.1213, -0.1785, -0.4739, -0.2935,
                                        -0.1715, -0.2214, -0.4290, -0.6645, -0.5292, -0.0227, -0.1752, -0.1616,
                                        -0.3258,  0.2725, -0.1213, -0.3028,  0.0065, -0.1135, -0.2444, -0.0786,
                                        -0.1471, -0.2499, -0.3083, -0.1046, -0.1604, -0.1245, -0.0130,  0.0275,
                                        -0.1689, -0.3215, -0.2388, -0.1091, -0.2070, -0.2346, -0.2325, -0.0447,
                                        -0.2822, -0.0886, -0.2100, -0.0973, -0.2107, -0.3271, -0.1520, -0.0941,
                                        -0.2882, -0.3017, -0.2254, -0.4124, -0.1171, -0.0857, -0.3209, -0.2435,
                                        -0.0546, -0.1269, -0.0447, -0.2978, -0.2079, -0.3572, -0.0906, -0.2392,
                                        -0.2157, -0.2214, -0.1016, -0.2315, -0.0957, -0.1174, -0.0928, -0.0844],
                                        grad_fn=<SelectBackward0>), 
                
                real_z=tensor([0.3720, 0.3978, 0.3700, 0.3742, 0.3862, 0.3039, 0.1904, 0.3325, 0.3671,
                                0.3993, 0.3978, 0.3661, 0.4056, 0.3765, 0.3904, 0.3819, 0.3721, 0.4034,
                                0.4151, 0.4031, 0.4291, 0.3784, 0.3899, 0.4134, 0.3894, 0.4118, 0.3662,
                                0.3593, 0.3943, 0.3398, 0.3851, 0.3587, 0.3748, 0.3722, 0.4164, 0.3729,
                                0.2191, 0.3898, 0.3980, 0.3659, 0.2919, 0.3763, 0.4508, 0.3770, 0.3804,
                                0.4358, 0.3660, 0.4103, 0.4256, 0.3449, 0.4093, 0.3359, 0.3857, 0.3938,
                                0.3084, 0.4031, 0.4642, 0.4004, 0.4183, 0.3543, 0.4116, 0.4083, 0.3765,
                                0.3798, 0.3813, 0.4130, 0.3712, 0.3957, 0.3851, 0.3634, 0.3813, 0.3812,
                                0.3715, 0.3765, 0.3924, 0.3703, 0.3770, 0.3067, 0.3963, 0.4178, 0.3923,
                                0.3698, 0.2426, 0.4257, 0.4541, 0.3852, 0.3772, 0.3761, 0.3784, 0.3948,
                                0.3847, 0.3598, 0.3867, 0.3803, 0.3739, 0.3677, 0.2631, 0.3683, 0.3923,
                                0.3953, 0.3542, 0.3935, 0.3591, 0.3525, 0.3874, 0.4056, 0.3770, 0.4045,
                                0.4505, 0.4134, 0.3543, 0.4140, 0.3928, 0.4151, 0.4337, 0.3952, 0.4376,
                                0.3854, 0.3659, 0.3718, 0.3755, 0.3543, 0.3789, 0.3865, 0.3634, 0.3929,
                                0.3979, 0.3975], grad_fn=<SigmoidBackward0>), 
                    
                fake_z=tensor([0.4792, 0.4613, 0.4411, 0.4736, 0.4698, 0.4952, 0.4803, 0.4297, 0.4026,
                                0.3774, 0.4600, 0.4326, 0.4533, 0.4300, 0.4425, 0.5004, 0.4434, 0.5182,
                                0.4287, 0.3623, 0.4688, 0.4222, 0.4509, 0.4606, 0.4475, 0.4805, 0.4403,
                                0.4484, 0.4624, 0.4420, 0.4601, 0.5050, 0.4572, 0.4666, 0.4900, 0.4332,
                                0.4472, 0.4311, 0.4680, 0.3767, 0.4719, 0.4454, 0.4668, 0.4731, 0.4987,
                                0.4878, 0.4324, 0.4892, 0.4224, 0.3607, 0.4603, 0.4171, 0.4478, 0.4063,
                                0.4463, 0.4640, 0.4643, 0.3676, 0.4118, 0.3581, 0.4697, 0.4555, 0.3837,
                                0.4271, 0.4572, 0.4449, 0.3944, 0.3397, 0.3707, 0.4943, 0.4563, 0.4597,
                                0.4193, 0.5677, 0.4697, 0.4249, 0.5016, 0.4717, 0.4392, 0.4804, 0.4633,
                                0.4379, 0.4235, 0.4739, 0.4600, 0.4689, 0.4967, 0.5069, 0.4579, 0.4203,
                                0.4406, 0.4727, 0.4484, 0.4416, 0.4421, 0.4888, 0.4299, 0.4779, 0.4477,
                                0.4757, 0.4475, 0.4189, 0.4621, 0.4765, 0.4284, 0.4251, 0.4439, 0.3983,
                                0.4708, 0.4786, 0.4205, 0.4394, 0.4864, 0.4683, 0.4888, 0.4261, 0.4482,
                                0.4116, 0.4774, 0.4405, 0.4463, 0.4449, 0.4746, 0.4424, 0.4761, 0.4707,
                                0.4768, 0.4789], grad_fn=<SigmoidBackward0>) """
            
            loss_D.backward()  # 反向传播，计算梯度，计算梯度是为了更新模型参数，使得模型参数能够更好的拟合数据，从而提高模型的性能，提高模型的性能是为了提高模型的泛化能力，提高模型的泛化能力是为了提高模型的预测准确率。
            self.opt_D.step()
        # ============================================
        #  Train Stu  训练学生模型，即训练生成器，生成器的目标是欺骗判别器。在训练生成器的时候, 它会调整内部的参数使得它生成的图片越来越真实。也就是它一直在优化使得它产生的数据如图片能够骗过D(判别器), 想要让D认为它生成的图片是真实的。对这些生成的图片，G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此这样的训练就叫做 对抗训练(adversarial training), 也叫做GAN。
        # ============================================
        if epoch % self.args.g_critic == 0: # g_critic 全称 generator critic，即生成器的评判者
            if self.args.adversarial_ouput_class == 16:
                loss_G = label_loss
                ## to fool Discriminator_l
                self.Discriminator.eval() # 评估模式，不启用 BatchNormalization 和 Dropout。
                pos_z = self.Discriminator(feat_fix_1) # 将教师模型的输出作为正样本，即真实样本。
                neg_z = self.Discriminator(feat_1) # 将学生模型的输出作为负样本，即生成样本。
                fake_z = torch.sigmoid(neg_z[:, -1]) # 将负样本的最后一列作为生成样本的概率。为什么是最后一列？因为最后一列是生成样本的概率？TODO
                ad_loss_1 = self.loss_dis(fake_z, torch.ones_like(fake_z)) # 计算生成样本(学生模型)的概率与真实样本的概率的BCEloss。通过和真实样本的概率比较，来判断生成样本的概率是不是越来越接近真实样本的概率。查看判别器这里的代码会发现是把生成器和zeros比较。又因为G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此对于生成器来说loss是其和1之间的差距，通过减少loss可以让其越来越和1(真实样本)接近，也就是越来越能骗过判别器。而对于判别器来说loss是其和0之间的差距，通过减少loss可以让其越来越和0(生成样本)接近，也就是越来越能区分生成样本和真实样本。判别器的目的是尽可能区分出真实样本和假样本，而生成器的目的是尽可能地欺骗判别器。
                # 验证测试，将fake_z和全0进行loss计算，预计会影响最后的准确率？
                # ad_loss = self.loss_dis(fake_z, torch.zeros_like(fake_z))
                
                # ds_loss = self.loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one # 计算生成样本(学生模型)和真实样本的交叉熵损失。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                l1_loss_1 = torch.norm(feat_1 - feat_fix_1, p=1) * 1. / len(feat_fix_1) # 计算生成样本与真实样本的L1范数。L1范数是指向量各个元素绝对值之和。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                loss_G_1 = loss_G + ad_loss_1 + l1_loss_1 # ad_loss全称adversarial loss，即对抗损失。ds_loss全称discriminator loss，即判别器损失。l1_loss全称L1 loss，即L1损失。

                self.Discriminator.eval() # 评估模式，不启用 BatchNormalization 和 Dropout。
                pos_z = self.Discriminator(feat_fix_2) # 将教师模型的输出作为正样本，即真实样本。
                neg_z = self.Discriminator(feat_2) # 将学生模型的输出作为负样本，即生成样本。
                fake_z = torch.sigmoid(neg_z[:, -1]) # 将负样本的最后一列作为生成样本的概率。为什么是最后一列？因为最后一列是生成样本的概率？TODO
                ad_loss_2 = self.loss_dis(fake_z, torch.ones_like(fake_z)) # 计算生成样本(学生模型)的概率与真实样本的概率的BCEloss。通过和真实样本的概率比较，来判断生成样本的概率是不是越来越接近真实样本的概率。查看判别器这里的代码会发现是把生成器和zeros比较。又因为G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此对于生成器来说loss是其和1之间的差距，通过减少loss可以让其越来越和1(真实样本)接近，也就是越来越能骗过判别器。而对于判别器来说loss是其和0之间的差距，通过减少loss可以让其越来越和0(生成样本)接近，也就是越来越能区分生成样本和真实样本。判别器的目的是尽可能区分出真实样本和假样本，而生成器的目的是尽可能地欺骗判别器。
                # 验证测试，将fake_z和全0进行loss计算，预计会影响最后的准确率？
                # ad_loss = self.loss_dis(fake_z, torch.zeros_like(fake_z))
                
                # ds_loss = self.loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one # 计算生成样本(学生模型)和真实样本的交叉熵损失。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                l1_loss_2 = torch.norm(feat_2 - feat_fix_2, p=1) * 1. / len(feat_fix_2) # 计算生成样本与真实样本的L1范数。L1范数是指向量各个元素绝对值之和。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                loss_G_2 = loss_G + ad_loss_2 + l1_loss_2 # ad_loss全称adversarial loss，即对抗损失。ds_loss全称discriminator loss，即判别器损失。l1_loss全称L1 loss，即L1损失。
                loss_G = 0.5 * (loss_G_1 + loss_G_2) # 两个生成器的损失相加。
            elif self.args.adversarial_ouput_class == 1:
                loss_G = label_loss
                ###################### to fool Discriminator_l
                self.Discriminator.eval() # 评估模式，不启用 BatchNormalization 和 Dropout。
                pos_z = self.Discriminator(tea_logits) # 将教师模型的输出作为正样本，即真实样本。
                neg_z = self.Discriminator(stu_logits) # 将学生模型的输出作为负样本，即生成样本。
                fake_z = torch.sigmoid(neg_z[:, -1]) # 将负样本的最后一列作为生成样本的概率。为什么是最后一列？因为最后一列是生成样本的概率？TODO
                ad_loss = self.loss_dis(fake_z, torch.ones_like(fake_z)) # 计算生成样本(学生模型)的概率与真实样本的概率的BCEloss。通过和真实样本的概率比较，来判断生成样本的概率是不是越来越接近真实样本的概率。查看判别器这里的代码会发现是把生成器和zeros比较。又因为G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此对于生成器来说loss是其和1之间的差距，通过减少loss可以让其越来越和1(真实样本)接近，也就是越来越能骗过判别器。而对于判别器来说loss是其和0之间的差距，通过减少loss可以让其越来越和0(生成样本)接近，也就是越来越能区分生成样本和真实样本。判别器的目的是尽可能区分出真实样本和假样本，而生成器的目的是尽可能地欺骗判别器。
                # 验证测试，将fake_z和全0进行loss计算，预计会影响最后的准确率？
                # ad_loss = self.loss_dis(fake_z, torch.zeros_like(fake_z))
                
                # ds_loss = self.loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one # 计算生成样本(学生模型)和真实样本的交叉熵损失。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                l1_loss = torch.norm(stu_logits - tea_logits, p=1) * 1. / len(tea_logits) # 计算生成样本与真实样本的L1范数。L1范数是指向量各个元素绝对值之和。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
                loss_G = loss_G + ad_loss + l1_loss # ad_loss全称adversarial loss，即对抗损失。ds_loss全称discriminator loss，即判别器损失。l1_loss全称L1 loss，即L1损失。

                ###################### to fool Discriminator_g
                self.Discriminator_g.eval()
                # tea_model_node_emb = feat_fix_1.detach() # 教师模型的节点embedding # tea_model_node_emb.shape=torch.Size([128, 16])
                # stu_model_node_emb = feat_1.detach() # 学生模型的节点embedding feat_joint.shape=torch.Size([128, 16])
                
                tea_model_node_emb = feat_joint_fix.detach() # 教师模型的节点embedding # tea_model_node_emb.shape=torch.Size([128, 16])
                stu_model_node_emb = feat_joint.detach() # 学生模型的节点embedding feat_joint.shape=torch.Size([128, 16])
                
                tea_sum = torch.sigmoid(tea_model_node_emb.mean(dim=0)).unsqueeze(-1) # tea_sum.shape=torch.Size([16, 1])
                neg_g = self.Discriminator_g(stu_model_node_emb, tea_sum)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = self.loss_dis(fake_g, torch.ones_like(fake_g))
                
                stu_sum = torch.sigmoid(stu_model_node_emb.mean(dim=0)).unsqueeze(-1) # stu_sum.shape=torch.Size([16, 1])
                neg_g = self.Discriminator_g(tea_model_node_emb, stu_sum)
                pos_g = self.Discriminator_g(stu_model_node_emb, stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = self.loss_dis(real_g, torch.zeros_like(real_g)) + self.loss_dis(fake_g, torch.ones_like(fake_g))
                # print('ad_gloss1:', ad_gloss1, 'ad_gloss2:', ad_gloss2) 
                loss_G = loss_G + ad_gloss1 + ad_gloss2

            # loss_G.backward() # 将loss_G反向传播，更新生成器的参数，使得生成器的损失函数loss_G最小化，即生成器生成的样本越来越接近真实样本。
            
            # loss = loss_G
            # loss = loss_reg + loss_kd + loss_reg_rec 
            loss = loss_reg + loss_kd + loss_reg_rec + loss_G
            if self.args.wandb:
                wandb.log({
                    "loss_reg": loss_reg.item(),
                    "loss_kd": loss_kd.item(),
                    "loss_reg_rec": loss_reg_rec.item(),
                    "loss_D": loss_D.item(),
                    "loss_G": loss_G.item(),
                    "loss": loss.item(),
                    "epoch": epoch,
                })

        # loss = loss_G
        # loss = loss_reg + loss_kd + loss_reg_rec 
        # loss = loss_reg + loss_kd + loss_reg_rec + loss_G
        if self.args.wandb:
            if loss == loss_G:
                wandb.config.stu_loss_type = "loss_G"
            elif loss == loss_reg + loss_kd + loss_reg_rec:
                wandb.config.stu_loss_type = "loss_reg+loss_kd+loss_reg_rec"
            elif loss == loss_reg + loss_kd + loss_reg_rec + loss_G:
                wandb.config.stu_loss_type = "loss_reg+loss_kd+loss_reg_rec+loss_G"
        
        loss.backward()
        self.optimizer_c1.step()
        self.optimizer_c.step()
        self.optimizer_f.step()
        self.optimizer_g.step()
        return loss_reg.item(), loss_kd.item()


    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")

        self.optimizer_g = torch.optim.Adam(self.model_g.parameters(), \
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), \
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_c = torch.optim.Adam(self.model_c.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_c1 = torch.optim.Adam(self.model_c1.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.opt_D = torch.optim.Adam(self.Discriminator.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        
        self.model_g.train() # model_g 全称是 model_gnn
        self.model_f.train() # model_f 全称是 model_fusion
        self.model_c.train() # model_c 全称是 model_classifier
        self.model_c1.train() # model_c1 全称是 model_classifier1
        
        self.loss_dis = torch.nn.BCELoss()
        self.loss_fcn = torch.nn.CrossEntropyLoss()

        if self.args.gnn_operator == 'mpnn':
            log_interval = 2
        else:
            log_interval = 10
            
        # epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        loss_list = []
        loss_list_test = []
        loss_list_kd = []
        loss_list_test_kd = []
        for epoch in range(self.args.epochs):
            scores = None
            if self.args.plot:
                if (epoch+1) % 50 == 0:
                    self.model_g.train(False)
                    self.model_f.train(False)
                    self.model_c.train(False)
                    cnt_test = 20
                    cnt_train = 100
                    # t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc = "Validation")
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g]*cnt_train)
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]

                        edge_index_1 = data["g1"].edge_index
                        edge_index_2 = data["g2"].edge_index
                        features_1 = data["g1"].x
                        features_2 = data["g2"].x
                        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
                        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
                        i_1 = data["g1"].i
                        i_2 = data["g2"].i

                        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1, i_1)
                        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2, i_2)

                        feat_joint = self.model_f(pooled_features_1_all, pooled_features_2_all)
                        prediction = self.model_c(feat_joint)
                        
                        """ print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 模型输入数据 edge_index_1.shape: {edge_index_1.shape} features_1.shape: {features_1.shape} batch_1.shape: {batch_1.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_g处理后，输出的pooled_features_1_all.shape: {pooled_features_1_all.shape} pooled_features_2_all.shape: {pooled_features_2_all.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_f处理后，输出的feat_joint.shape: {feat_joint.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_c处理后，最终返回的预测结果prediction.shape: {prediction.shape}，目标target.shape: {target.shape}')
                        AIDS700nef 数据集打印结果
                        模型输入数据 edge_index_1.shape: torch.Size([2, 1600]) features_1.shape: torch.Size([800, 29]) batch_1.shape: torch.Size([800])
                        经过model_g处理后，输出的pooled_features_1_all.shape: torch.Size([100, 112]) pooled_features_2_all.shape: torch.Size([100, 112])
                        经过model_f处理后，输出的feat_joint.shape: torch.Size([100, 16])
                        经过model_c处理后，最终返回的预测结果prediction.shape: torch.Size([100])，目标target.shape: torch.Size([100]) """
                        
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
                        # t.update(cnt_train)
                    
                    # t.close()
                    loss_list_test.append(scores.mean().item())
                    if self.args.wandb:
                        wandb.log({
                            "val_loss": scores.mean().item(),
                            "epoch": epoch+1,
                            })
                    self.model_g.train(True)
                    self.model_f.train(True)
                    self.model_c.train(True)
            
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            loss_sum_kd = 0
            # for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
            for index, batch_pair in enumerate(batches):
                if self.args.use_adversarial == 1:
                    # print('[EGSC-KD/src/egsc_kd.py] 准备执行 process_batch_adversarial 函数')
                    loss_score, loss_score_kd = self.process_batch_adversarial(batch_pair, epoch)
                else:
                    # print('[EGSC-KD/src/egsc_kd.py] 准备执行 process_batch 函数')
                    loss_score, loss_score_kd = self.process_batch(batch_pair, epoch)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
                loss_sum_kd = loss_sum_kd + loss_score_kd
            loss = loss_sum / main_index
            loss_kd = loss_sum_kd / main_index
            # epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
            loss_list.append(loss)
            loss_list_kd.append(loss_kd)

            if epoch == 0 or ((epoch+1) % log_interval) == 0:
                if scores is not None:
                    print('[Fit_Train] Epoch:{} Train Loss:{} Loss_kd:{} Valid Loss:{}'.format(epoch+1, round(loss,5), round(loss_kd,5), scores.mean().item())) # 打印每一轮的loss
                else:
                    print('[Fit_Train] Epoch:{} Train Loss:{} Loss_kd:{}'.format(epoch+1, round(loss,5), round(loss_kd,5))) # 打印每一轮的loss # print validation loss
            if ((epoch+1) % self.args.test_interval) == 0: # 每test_interval轮进行一次测试Test
                self.score(epoch+1) # print Test result
            if self.args.wandb:
                wandb.log({
                    "train_loss": loss,
                    "train_loss_kd": loss_kd,
                    "epoch": epoch+1,
                    })


        if self.args.plot:
            filename_meta = 'figs/' + self.args.dataset
            filename_meta += '_' + self.args.gnn_operator 
            if self.args.diffpool:
                filename_meta += '_diffpool'
            if self.args.histogram:
                filename_meta += '_hist'
            filename_meta = filename_meta + str(self.args.epochs) + self.args.mode

            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")

            plt.legend()
            filename = filename_meta + '_mse.pdf'
            plt.savefig(filename)
            plt.close()
            plt.plot([*range(0, self.args.epochs, 1)], loss_list_kd, label="Train KD")

            plt.legend()
            filename = filename_meta + '_kd.pdf'
            plt.savefig(filename)
            plt.close()

            loss_list_kd_array=np.array(loss_list_kd)
            np.save(filename_meta + '_kd.npy',loss_list_kd_array)
            loss_list_array=np.array(loss_list)
            np.save(filename_meta + '_mse_train.npy',loss_list_array)
            loss_list_test_array=np.array(loss_list_test)
            np.save(filename_meta + '_mse_test.npy',loss_list_test_array)

    def score(self, epoch=None):
        """
        Scoring.
        """
        print("**************************************** Model evaluation (Test) ****************************************")
        self.model_g.eval()
        self.model_f.eval()
        self.model_c.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行 score 函数，len(self.testing_graphs): {len(self.testing_graphs)}, len(self.training_graphs): {len(self.training_graphs)}') # len(self.testing_graphs): 140, len(self.training_graphs): 560
        # t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs)) # len(self.testing_graphs)*len(self.training_graphs) = 140*560 = 78400

        # 记录当前时间戳
        start_time = time.time()
        for i, g in enumerate(self.testing_graphs): # self.testing_graphs 为一个长度为 140 的列表，每个元素都是一个图
            # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行 score 函数，i: {i}, g: {g}')
            source_batch = Batch.from_data_list([g]*len(self.training_graphs)) # [g]*len(self.training_graphs) 为一个长度为 560 的列表，每个元素都是 g (当前循环到的测试集里的一个图). source_batch代表的是一个批次的图，每个图都是 g，批次大小为 560，即一次性处理 560 个图
            target_batch = Batch.from_data_list(self.training_graphs) # self.training_graphs 为一个长度为 560 的列表，每个元素都是训练集里的一个图，target_batch代表的是一个批次的图，每个图都是 self.training_graphs 中的一个图，批次大小为 560，即一次性处理 560 个图
            # source_batch 是560个相同的图(当前循环到的测试集里的一个图)，target_batch 是560个不同的图(训练集里的560个图)
            
            # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行 score 函数，source_batch: {source_batch}, target_batch: {target_batch}')
            # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行 score 函数，source_batch["i"]: {source_batch["i"]}, target_batch["i"]: {target_batch["i"]}')
            """ source_batch: DataBatch(edge_index=[2, 10080], i=[560], x=[5600, 29], num_nodes=5600, batch=[5600], ptr=[561])
            target_batch: DataBatch(edge_index=[2, 9898], i=[560], x=[4991, 29], num_nodes=4991, batch=[4991], ptr=[561])
            source_batch["i"]: tensor([560, 560, ..., 560]) # i=0时即第一个循环为560个560，下一个循环会变为560个561
            target_batch["i"]: tensor([  0,   1,   2, ..., 559]) #从 0 到 559 的 560 个数 每个循环都是一样的""" 
            
            data = self.transform((source_batch, target_batch))
            # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行 score 函数，data: {data}')
            """ 第一个循环i=0的 data:
            data: {
                'g1': DataBatch(edge_index=[2, 10080], i=[560], x=[5600, 29], num_nodes=5600, batch=[5600], ptr=[561]), 
                'g2': DataBatch(edge_index=[2, 9898], i=[560], x=[4991, 29], num_nodes=4991, batch=[4991], ptr=[561]), 
                'target': tensor([0.3012, 0.4308, 0.5318, ..., 0.2828, 0.2828, 0.6065]), 
                'target_ged': tensor([12.,  8.,  6.,  8.,  ...., 12.,  5.])}  符合预期 因为self.ged[560][0]=12.0 self.ged[560][1]=8.0

            验证：.conda/envs/Efficient_Graph_Similarity_Computation/lib/python3.9/site-packages/torch_geometric/datasets/ged_dataset.py的print信息，
                self.ged[560][0]=12.0, 即索引为560的图与索引为0的图的ged为12.0
                self.ged[560][1]=8.0,  即索引为560的图与索引为1的图的ged为8.0
                self.ged[560][559]=5.0, 即索引为560的图与索引为559的图的ged为5.0
                 
                self.ged[561][0]=14.0, 即索引为561的图与索引为0的图的ged为14.0
                self.ged[561][1]=11.0, 即索引为561的图与索引为1的图的ged为11.0
                self.ged[561][559]=13.0 即索引为561的图与索引为559的图的ged为13.0
            
            第2个循环i=1时的 data:
            'target': tensor([0.1738, 0.2307, , 0.3012, 0.3012, 0.1969]) # 一共560个数，表示第i个测试图与训练集中的所有560个图的相似度 ⭐️
            'target_ged': tensor([14., 11., 11., ...,  9., 13.])} 符合预期 因为self.ged[561][0]=14.0 self.ged[561][1]=11.0 self.ged[561][559]=13.0
            """
            
        
            target = data["target"] 
            ground_truth[i] = target # 表示第i个测试图与训练集中的所有560个图的相似度，即560个数 （正则化）
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged # 表示第i个测试图与训练集中的所有560个图的相似度，即560个数 （没有正则化）

            edge_index_1 = data["g1"].edge_index.to(self.device)
            edge_index_2 = data["g2"].edge_index.to(self.device)
            features_1 = data["g1"].x.to(self.device)
            features_2 = data["g2"].x.to(self.device)
            batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
            batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
            batch_1 = batch_1.to(self.device)
            batch_2 = batch_2.to(self.device)
            i_1 = data["g1"].i.to(self.device)
            i_2 = data["g2"].i.to(self.device)

            pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
            pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

            feat_joint = self.model_f(pooled_features_1_all, pooled_features_2_all).to(self.device)

            prediction = self.model_c(feat_joint).cpu()

            # print("target.device: ", target.device, "prediction.device: ", prediction.device)
            prediction_mat[i] = prediction.detach().numpy()

            # print(f'prediction.shape: {prediction.shape}, target.shape: {target.shape}') # prediction.shape: torch.Size([560]), target.shape: torch.Size([560])
            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy() # 计算第i个测试图与训练集中的所有560个图的相似度的MSE，即将所有560个数进行MSE计算

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

            # t.update(len(self.training_graphs)) # update表示更新进度条，len(self.training_graphs)表示更新的长度，即更新的进度条的长度
            
            # 一次循环对比了第i个测试图和所有560个训练图，即560个数。一共140个测试图，所以一共对比了140*560=78400个ged值。每个ged值代表了一对图的相似度，即一共对比了78400对图的相似度。
            # 如果所有循环一共耗时6秒，那么每个ged值耗时6/78400=0.000076秒，即每秒计算13000个ged值，即每秒计算13000对图的相似度。
            # 注意：如果把计算loss的代码注释掉，

        # 记录循环结束时的时间戳
        end_time = time.time()
        # 计算循环耗时，一共对比了140*560=78400个ged值
        # print("score time cost: ", end_time - start_time) # CAM-HPC CPU： 6.491062164306641秒 ； CAM-HPC GPU： 3.4038896560668945秒
        if epoch is None:
            print('[Score] score_time_cost:', end_time - start_time, 'seconds')
        else:
            print('[Fit_Test] Epo:', epoch, ', score_time_cost:', end_time - start_time, ' seconds')
        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        if epoch is not None:
            self.print_evaluation_epoch(epoch)
        else:
            self.print_evaluation()
        if self.args.wandb:
            wandb.log({
                "rho": self.rho,
                "tau": self.tau,
                "prec_at_10": self.prec_at_10,
                "prec_at_20": self.prec_at_20,
                "model_error": self.model_error,
                # "run index": i
                })

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")

    def print_evaluation_epoch(self, epoch):
        """
        Printing the error rates with epoch info during fit stage.
        """
        print("[Fit_Test] Epo:" + str(epoch) + ", mse(10^-3):" + str(round(self.model_error*1000, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", Spearman's rho:" + str(round(self.rho, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", Kendall's tau:" + str(round(self.tau, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", p@10:" + str(round(self.prec_at_10, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", p@20:" + str(round(self.prec_at_20, 5)) + ".")