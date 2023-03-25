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
        
        # if torch.cuda.device_count() > 1:  
        #         self.model_g = torch.nn.DataParallel(self.model_g) 
        #         self.model_f = torch.nn.DataParallel(self.model_f)
        #         self.model_c = torch.nn.DataParallel(self.model_c)
        #         self.model_c1 = torch.nn.DataParallel(self.model_c1)
        #         # self.model_g_fix = torch.nn.DataParallel(self.model_g_fix)
        #         self.Discriminator = torch.nn.DataParallel(self.Discriminator)
    
        self.get_parameter_number(self.model_g) 
        # self.get_parameter_number(self.model_f)
        # self.get_parameter_number(self.model_c)
        # self.get_parameter_number(self.model_c1)
        self.get_parameter_number(self.model_g_fix)
         
        self.model_g.to(self.device)
        self.model_f.to(self.device)
        self.model_c.to(self.device)
        self.model_c1.to(self.device)
        self.model_g_fix.to(self.device)
        self.Discriminator.to(self.device)
        # self.Discriminator_e.to(self.device)
        self.Discriminator_g.to(self.device)
        
        self.loss_RkdDistance = RkdDistance()
        self.loss_RKdAngle = RKdAngle()

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'model params，total_num: {total_num}，trainable_num: {trainable_num}')

    def load_model(self):
        if self.args.checkpoint:
            PATH_g = self.args.checkpoint
        else:
            PATH_g = '../Checkpoints/G_EarlyFusion_Disentangle_' +str(self.args.dataset) +'_gin'+'_checkpoint.pth'
        
        print('Loading Teacher Model from {}'.format(PATH_g))
        
        # print('self.model_g_fix: ', self.model_g_fix, '\nteacher model:', torch.load(PATH_g, map_location=self.device))
        
        self.model_g_fix.load_state_dict(torch.load(PATH_g, map_location=self.device), strict=False) 
        
        # self.model_g_fix.load_state_dict(torch.load(PATH_g), strict=False) # because use light model, so we need to load the model with strict=False
        
        print('Model Loaded')
        
    def process_dataset(self):
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.nged_matrix = self.training_graphs.norm_ged 
        self.ged_matrix = self.training_graphs.ged 

        self.real_data_size = self.nged_matrix.size(0)
        
        if self.args.synth:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)  
            
            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat((self.nged_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat((torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))
        
        if self.training_graphs[0].x is None:
            max_degree = 0 
            for g in self.training_graphs + self.testing_graphs + (self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0: 
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
        
        # labeling of synth data according to real data format     
        if self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g = one_hot_degree(g)
                g.i = g.i + real_data_size 
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size    
        self.number_of_labels = self.training_graphs.num_features 
        pass

    def create_batches(self):
        # print(f"self.args.synth = {self.args.synth}") # self.args.synth = False
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader)) 

    def transform(self, data):
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        
        normalized_ged = self.nged_matrix[ 
            data[0]["i"].reshape(-1).tolist(),
            data[1]["i"].reshape(-1).tolist()
            ].tolist() 
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        
        ged = self.ged_matrix[
            data[0]["i"].reshape(-1).tolist(),
            data[1]["i"].reshape(-1).tolist()
            ].tolist() 
        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()

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

        loss = loss_reg + loss_kd + loss_reg_rec 
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
        target = data["target"].to(self.device) 

        edge_index_1 = data["g1"].edge_index.to(self.device) 
        edge_index_2 = data["g2"].edge_index.to(self.device)
        features_1 = data["g1"].x.to(self.device)
        features_2 = data["g2"].x.to(self.device)
        test1 = hasattr(data["g1"], 'batch') # test1: True
        test2 = hasattr(data["g2"], 'batch') # test2: True
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
        label_loss = loss_reg

        pooled_features_1 = self.model_g(edge_index_1, features_1, batch_1, i_1).to(self.device)
        pooled_features_2 = self.model_g(edge_index_2, features_2, batch_2, i_2).to(self.device)

        feat_joint = self.model_f(pooled_features_1, pooled_features_2)
        feat_joint_1 = self.model_f(pooled_features_1, pooled_features_1)
        feat_joint_2 = self.model_f(pooled_features_2, pooled_features_2)
        feat_1 = feat_joint-feat_joint_1
        feat_2 = feat_joint-feat_joint_2
        
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
        #  Train Dis
        # ============================================
        if epoch % self.args.d_critic == 0:
            loss_D = 0
            if self.args.adversarial_ouput_class == 1:
                ###################### distinguish by Dl
                # test option1: use prediction as stu_logits and target as tea_logits
                stu_logits = prediction.unsqueeze(1)
                tea_logits = target.unsqueeze(1)
                
                self.Discriminator.train() 
                stu_logits = stu_logits.detach() 
                pos_z = self.Discriminator(tea_logits) 
                neg_z = self.Discriminator(stu_logits) 
                real_z = torch.sigmoid(pos_z[:, -1]) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                
                ad_loss = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) 
                tea_model_node_emb = feat_joint_fix.detach()
                stu_model_node_emb = feat_joint.detach()
                
                self.Discriminator_g.train()
                tea_sum = torch.sigmoid(tea_model_node_emb.mean(dim=0)).unsqueeze(-1) 
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
            
                self.Discriminator.train() 
                stu_logits_1 = stu_logits_1.detach() 
                pos_z = self.Discriminator(tea_logits_1) 
                neg_z = self.Discriminator(stu_logits_1) 
                real_z = torch.sigmoid(pos_z[:, -1]) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                
                ad_loss_1 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) 
                
                stu_logits_2 = stu_logits_2.detach() 
                pos_z = self.Discriminator(tea_logits_2) 
                neg_z = self.Discriminator(stu_logits_2) 
                real_z = torch.sigmoid(pos_z[:, -1]) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                ad_loss_2 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) 
                loss_D = 0.5 * (ad_loss_1 + ad_loss_2) 
            else:
                raise ValueError("error: adversarial_ouput_class should be 1 or 16")

            loss_D.backward()  
            self.opt_D.step()
        # ============================================
        #  Train Stu
        # ============================================
        if epoch % self.args.g_critic == 0: 
            if self.args.adversarial_ouput_class == 16:
                loss_G = label_loss
                ## to fool Discriminator_l
                self.Discriminator.eval() 
                pos_z = self.Discriminator(feat_fix_1) 
                neg_z = self.Discriminator(feat_1) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                ad_loss_1 = self.loss_dis(fake_z, torch.ones_like(fake_z)) 
                l1_loss_1 = torch.norm(feat_1 - feat_fix_1, p=1) * 1. / len(feat_fix_1) 
                loss_G_1 = loss_G + ad_loss_1 + l1_loss_1 

                self.Discriminator.eval() 
                pos_z = self.Discriminator(feat_fix_2) 
                neg_z = self.Discriminator(feat_2) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                ad_loss_2 = self.loss_dis(fake_z, torch.ones_like(fake_z)) 
                l1_loss_2 = torch.norm(feat_2 - feat_fix_2, p=1) * 1. / len(feat_fix_2) 
                loss_G_2 = loss_G + ad_loss_2 + l1_loss_2 
                loss_G = 0.5 * (loss_G_1 + loss_G_2) 
            elif self.args.adversarial_ouput_class == 1:
                loss_G = label_loss
                ###################### to fool Discriminator_l
                self.Discriminator.eval() 
                pos_z = self.Discriminator(tea_logits) 
                neg_z = self.Discriminator(stu_logits) 
                fake_z = torch.sigmoid(neg_z[:, -1]) 
                ad_loss = self.loss_dis(fake_z, torch.ones_like(fake_z)) 
                l1_loss = torch.norm(stu_logits - tea_logits, p=1) * 1. / len(tea_logits) 
                loss_G = loss_G + ad_loss + l1_loss 

                ###################### to fool Discriminator_g
                self.Discriminator_g.eval()
                tea_model_node_emb = feat_joint_fix.detach() 
                stu_model_node_emb = feat_joint.detach() 
                
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
        
        
        self.model_g.train() # model_g  model_gnn
        self.model_f.train() # model_f  model_fusion
        self.model_c.train() # model_c  model_classifier
        self.model_c1.train() # model_c1  model_classifier1
        
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
                    loss_score, loss_score_kd = self.process_batch_adversarial(batch_pair, epoch)
                else:
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
                    print('[Fit_Train] Epoch:{} Train Loss:{} Loss_kd:{} Valid Loss:{}'.format(epoch+1, round(loss,5), round(loss_kd,5), scores.mean().item())) 
                else:
                    print('[Fit_Train] Epoch:{} Train Loss:{} Loss_kd:{}'.format(epoch+1, round(loss,5), round(loss_kd,5))) 
            if ((epoch+1) % self.args.test_interval) == 0:
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
        start_time = time.time()
        for i, g in enumerate(self.testing_graphs): 
            source_batch = Batch.from_data_list([g]*len(self.training_graphs)) 
            target_batch = Batch.from_data_list(self.training_graphs) 

            data = self.transform((source_batch, target_batch))
        
            target = data["target"] 
            ground_truth[i] = target 
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged 

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
            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy() 

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

        end_time = time.time()
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