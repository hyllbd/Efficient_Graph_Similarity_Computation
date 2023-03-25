import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt

from model import EGSCT_generator, EGSCT_classifier

import pdb
import wandb
import time
torch.set_printoptions(precision=5, sci_mode=False, linewidth=2000)
np.set_printoptions(precision=5, suppress=True)

class EGSCTrainer(object):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.device = torch.device('cuda:{}'.format(self.args.cuda_id) if torch.cuda.is_available() else 'cpu')
        print('[EGSCTrainer] Using device:', self.device)
        self.process_dataset()
        self.setup_model()
        self.best_rho = 0
        self.best_tau = 0
        self.best_prec_at_10 = 0
        self.best_prec_at_20 = 0
        self.best_model_error = float('inf')


    def setup_model(self):
        """
        Creating a EGSC.
        """
        self.model_g = EGSCT_generator(self.args, self.number_of_labels)
        self.model_c = EGSCT_classifier(self.args, self.number_of_labels)
        self.model_g.to(self.device)
        self.model_c.to(self.device)
        
        self.get_parameter_number(self.model_g)
        self.get_parameter_number(self.model_c)


    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model:{model.__class__.__name__} Parameters total_num: {total_num}，trainable_num: {trainable_num}')

    def save_model(self):
        """
        Saving a EGSC.
        """
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_g.state_dict(), PATH_g)

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_c.state_dict(), PATH_c)
        
        if self.args.wandb:
            wandb.log({
                "saved_model_c_path": PATH_c,
                "saved_model_g_path": PATH_g,
                })
        print('Model Saved, PATH_c:', PATH_c, ', PATH_g:', PATH_g)

    def load_model(self):
        """
        Loading a EGSC.
        """
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_g.load_state_dict(torch.load(PATH_g))

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_c.load_state_dict(torch.load(PATH_c))
        print('Model Loaded')
        
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        print(f'[process_dataset] self.training_graphs[0] = {self.training_graphs[0]}') 
        # AIDS: self.training_graphs[0] = Data(edge_index=[2, 18], i=[1], x=[10, 29], num_nodes=10)
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        
        # self.testing_graphs.norm_ged
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

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        # print('self.args.batch_size:', self.args.batch_size)
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        

        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()

        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()

        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data. 
        """
        self.optimizer.zero_grad()
        
        data = self.transform(data)
        target = data["target"].to(self.device)
        
        edge_index_1 = data["g1"].edge_index.to(self.device)
        edge_index_2 = data["g2"].edge_index.to(self.device)
        features_1 = data["g1"].x.to(self.device)
        features_2 = data["g2"].x.to(self.device)
        batch_1 = data["g1"].batch.to(self.device) if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).to(self.device)
        batch_2 = data["g2"].batch.to(self.device) if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).to(self.device)
        i_1 = data["g1"].i.to(self.device)
        i_2 = data["g2"].i.to(self.device)
        prediction = self.model_c(self.model_g(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2))

        # print('[process_batch] prediction.device:', prediction.device, 'target.device:', target.device)
        # print('[process_batch] prediction.shape:', prediction.shape, 'target.shape:', target.shape) # [batch_size]
        # print('[process_batch] prediction[:20]:', prediction[:20], '\ntarget[:20]:', target[:20])
        loss = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        prediction = prediction.cpu().detach()
        target = target.cpu().detach()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def fit(self):
        """
        Training a model.
        """
        print("\n**************************************** Model training & validation. ****************************************\n")
        self.optimizer = torch.optim.Adam([{'params': self.model_g.parameters()}, {'params': self.model_c.parameters()}],\
         lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model_g.train()
        self.model_c.train()
        if self.args.gnn_operator == 'mpnn':
            log_interval = 2
        else:
            log_interval = 10
        loss_list = []
        loss_list_test = []
        epoch_list_test = []
        for epoch in range(self.args.epochs):
            scores = None
            if self.args.plot: 
                if (epoch+1) % 10 == 0: 
                    self.model_g.train(False) 
                    self.model_c.train(False) 
                    cnt_test = 20 
                    cnt_train = 100 
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()): 
                        source_batch = Batch.from_data_list([g]*cnt_train) 
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle()) 
                        data = self.transform((source_batch, target_batch)) 
                        target = data["target"]
                        
                        edge_index_1 = data["g1"].edge_index.to(self.device)
                        edge_index_2 = data["g2"].edge_index.to(self.device)
                        features_1 = data["g1"].x.to(self.device)
                        features_2 = data["g2"].x.to(self.device)
                        batch_1 = data["g1"].batch.to(self.device) if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).to(self.device)
                        batch_2 = data["g2"].batch.to(self.device) if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).to(self.device)
                        i_1 = data["g1"].i.to(self.device)
                        i_2 = data["g2"].i.to(self.device)

                        prediction = self.model_c(self.model_g(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2))
                        
                        prediction = prediction.cpu().detach()
                        target = target.cpu().detach()
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
                    
                    loss_list_test.append(scores.mean().item())
                    epoch_list_test.append(epoch)
                    if self.args.wandb:
                        wandb.log({
                            "val_loss": scores.mean().item(),
                            "epoch": epoch+1,
                            })
                    
                    self.model_g.train(True) 
                    self.model_c.train(True)
            
            batches = self.create_batches() 
            main_index = 0 
            loss_sum = 0 
            # for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
            for index, batch_pair in enumerate(batches): 
                loss_score = self.process_batch(batch_pair) 
                main_index = main_index + batch_pair[0].num_graphs 
                loss_sum = loss_sum + loss_score 
            loss = loss_sum / main_index 
            loss_list.append(loss) 
            
            if epoch == 0 or ((epoch+1) % log_interval) == 0:
                if scores is not None:
                    print('[Fit_Train] Epoch:{} Train Loss:{} Valid Loss:{}'.format(epoch+1, round(loss,5), scores.mean().item()))
                else:
                    print('[Fit_Train] Epoch:{} Train Loss:{}'.format(epoch+1, round(loss,5))) 
            if ((epoch+1) % self.args.test_interval) == 0: 
                self.score(epoch+1) # print Test result
            if self.args.wandb:
                wandb.log({
                    "train_loss": loss,
                    "epoch": epoch+1,
                    })
        # print validation loss
        for i in range(len(loss_list_test)):
            print('[Fit_Valid] Epoch:{}, Loss: {:.5f}'.format(epoch_list_test[i]+1, loss_list_test[i]))
            
        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation") # [*range(0, self.args.epochs, 10)] 0, 10, 20, 30, .... self.args.epochs
            plt.ylim([0, 0.01])
            plt.legend()
            
            # filename = 'figs/' + self.args.dataset
            # filename += '_' + self.args.gnn_operator 
            # filename = filename + str(self.args.epochs) + '.pdf'
            filename = 'figs/' + self.args.dataset + '_' + self.args.gnn_operator + '_' + str(self.args.epochs) + '_' + str(self.args.batch_size) + '_' + str(self.args.learning_rate) + '.pdf'
            plt.savefig(filename)
            

    def score(self, epoch=None):
        """
        Scoring.
        """
        # if epoch is None:
        print("**************************************** Model evaluation (Test) ****************************************")

        self.model_g.eval() 
        self.model_c.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        
        rho_list = [] 
        tau_list = [] 
        prec_at_10_list = [] 
        prec_at_20_list = [] 
        
        # t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs))

        start_time = time.time()
        for i, g in enumerate(self.testing_graphs): 
            # print('[Fit_Test] enumerate(self.testing_graphs) current i =', i)
            source_batch = Batch.from_data_list([g]*len(self.training_graphs)) 
            target_batch = Batch.from_data_list(self.training_graphs) 
            # print('[teacher-score] len(source_batch) = ', len(source_batch), 'len(target_batch) = ', len(target_batch)) # 560, 560
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            # print('[teacher-score]  data["target"].device = ',  data["target"].device, ', data["target_ged"].device = ', data["target_ged"].device) # cpu, cpu
            ground_truth[i] = target.cpu()
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged

            edge_index_1 = data["g1"].edge_index.to(self.device)
            edge_index_2 = data["g2"].edge_index.to(self.device)
            features_1 = data["g1"].x.to(self.device)
            features_2 = data["g2"].x.to(self.device)
            batch_1 = data["g1"].batch.to(self.device) if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).to(self.device)
            batch_2 = data["g2"].batch.to(self.device) if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).to(self.device)
            i_1 = data["g1"].i.to(self.device)
            i_2 = data["g2"].i.to(self.device)
            prediction = self.model_c(self.model_g(edge_index_1, features_1, batch_1, i_1, edge_index_2, features_2, batch_2, i_2))
            prediction_mat[i] = prediction.cpu().detach().numpy()
            target = target.to(self.device)
            scores[i] = F.mse_loss(prediction, target, reduction='none').cpu().detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

            # print('\n[Fit_Test] i =', i, ', prediction[:9]=', prediction_mat[i][:9], ', prediction.shape =', prediction_mat[i].shape, '\ntarget[:9] =', ground_truth[i][:9], ', target.shape =', ground_truth[i].shape, '\nscores[i][:9]=', scores[i][:9], ', scores.shape =', scores[i].shape)
            # print('p@20:', prec_at_20_list[-1], ', p@10:', prec_at_10_list[-1], ', rho:', rho_list[-1], ', tau:', tau_list[-1])
            # print('[Fit_Test] i =', i, ', scores[i] =', scores[i], ', prediction_mat[i] =', prediction_mat[i], ', ground_truth[i] =', ground_truth[i], ', ground_truth_ged[i] =', ground_truth_ged[i])


            # t.update(len(self.training_graphs))
        end_time = time.time()
        score_time_cost = end_time - start_time
        if epoch is None:
            print('[Score] score_time_cost:', score_time_cost, 'seconds')
        else:
            print('[Fit_Test] Epo:', epoch, ', score_time_cost:', score_time_cost, ' seconds')

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
                "score_time_cost": score_time_cost})


    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3):" + str(round(self.model_error*1000, 5)) + ".")
        print("Spearman's rho:" + str(round(self.rho, 5)) + ".")
        print("Kendall's tau:" + str(round(self.tau, 5)) + ".")
        print("p@10:" + str(round(self.prec_at_10, 5)) + ".")
        print("p@20:" + str(round(self.prec_at_20, 5)) + ".")

    def print_evaluation_epoch(self, epoch):
        """
        Printing the error rates with epoch info during fit stage.
        """
        print("[Fit_Test] Epo:" + str(epoch) + ", mse(10^-3):" + str(round(self.model_error*1000, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", Spearman's rho:" + str(round(self.rho, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", Kendall's tau:" + str(round(self.tau, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", p@10:" + str(round(self.prec_at_10, 5)) + ".")
        print("[Fit_Test] Epo:" + str(epoch) + ", p@20:" + str(round(self.prec_at_20, 5)) + ".")