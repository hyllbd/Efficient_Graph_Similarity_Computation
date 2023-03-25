import torch
from torch import Tensor as Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Linear
from torch.nn.init import _calculate_correct_fan, calculate_gain
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MessagePassing
import math

def map_x_to_u(x, batch_info): 
    """ map the node features to the right row of the initial local context.""" 
    # x = data.x
    num_nodes = x.shape[0]
    u = x.new_zeros((num_nodes, batch_info['n_colors'])) # u: num_nodes x n_colors e.g. 39 x 23
    u.scatter_(1, batch_info['coloring'], 1) 
    u = u[..., None] 
    
    u_x = u.new_zeros((u.shape[0], u.shape[1], x.shape[1])) # num_nodes x n_colors x n_features

    n_features = x.shape[1] 
    coloring = batch_info['coloring']       # num_nodes x 1
    expanded_colors = coloring[..., None].expand(-1, -1, n_features) # num_nodes x 1 x n_features

    u_x = u_x.scatter_(dim=1, index=expanded_colors, src=x[:, None, :]) # num_nodes x n_colors x n_features 
    
    # print('\n[map_x_to_u] u_x.shape:', u_x.shape, ', u.shape:', u.shape, ', expanded_colors.shape:', expanded_colors.shape)
    # u_x.shape:  torch.Size([num_nodes, n_colors, n_features]) , u.shape:  torch.Size([num_nodes, n_colors, 1]) , expanded_colors.shape: torch.Size([39, 1, 1])
    
    u = torch.cat((u, u_x), dim=2) # u.shape: torch.Size([num_nodes, n_colors, 2])
    # print('[map_x_to_u] final return u.shape:', u.shape)
    return u

# def create_batch_info(data, edge_counter): 
def create_batch_info(edge_index, x, batch, graph_index, edge_attr, edge_counter): 
    """ Compute some information about the batch that will be used by SMP."""
    # x, edge_index, batch, batch_size = data.x, edge_index, data.batch, data.num_graphs
    
    # print(f'data.y.shape: {data.y.shape}') # edge_index.shape: torch.Size([128])
    # batch_size = data.y.shape[0]
    batch_size = graph_index.numel()
    # print(f'[create_batch_info] batch_size: {batch_size}')
    # Compute some information about the batch
    
    # Count the number of nodes in each graph
    unique, n_per_graph = torch.unique(batch, return_counts=True)
    n_batch = torch.zeros_like(batch, dtype=torch.float) # num_nodes x 1

    for value, n in zip(unique, n_per_graph):
        n_batch[batch == value] = n.float()  

    # Count the average number of edges per graph
    num_nodes = x.shape[0]
    dummy = x.new_ones((num_nodes, 1))
    average_edges = edge_counter(dummy, edge_index, batch, batch_size)

    # Create the coloring if it does not exist yet
    # if not hasattr(data, 'coloring'):
    #     data.coloring = x.new_zeros(num_nodes, dtype=torch.long)
    #     # for i in range(data.num_graphs):
    #     for i in range(batch_size):
    #         data.coloring[data.batch == i] = torch.arange(n_per_graph[i], device=x.device)
    #     data.coloring = data.coloring[:, None]
    coloring = x.new_zeros(num_nodes, dtype=torch.long)
    # for i in range(data.num_graphs):
    for i in range(batch_size):
        coloring[batch == i] = torch.arange(n_per_graph[i], device=x.device)
    coloring = coloring[:, None]
    n_colors = torch.max(coloring) + 1  # Indexing starts at 0 

    mask = torch.zeros(num_nodes, n_colors, dtype=torch.bool, device=x.device)
    for value, n in zip(unique, n_per_graph):
        mask[batch == value, :n] = True

    # Aggregate into a dict
    batch_info = {'num_nodes': num_nodes,      
                #   'num_graphs': data.num_graphs, 
                  'num_graphs': batch_size,         
                  'batch': batch,              
                  'n_per_graph': n_per_graph,       
                  'n_batch': n_batch[:, None, None].float(),  
                  'average_edges': average_edges[:, :, None], 
                  'coloring': coloring,        
                  'n_colors': n_colors,             
                  'mask': mask      # Used because of batching - it tells which entries of u are not used by the graph 
                  }
    return batch_info

class BatchNorm(nn.Module):
    def __init__(self, channels: int, use_x: bool):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.use_x = use_x

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, u):
        if self.use_x:
            return self.bn(u)
        else:
            return self.bn(u.transpose(1, 2)).transpose(1, 2)

class EdgeCounter(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, batch, batch_size): 
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return global_mean_pool(n_edges, batch, batch_size)[batch] # global_mean_pool

class XtoGlobal(Linear):
    def forward(self, x: Tensor, batch_info: dict, method='mean'):
        """ x: (num_nodes, in_features). """
        g = pooling(x, batch_info, method)  # bs, N, in_feat or bs, in_feat
        return self.lin.forward(g)

def pooling(x: torch.Tensor, batch_info, method):
    if method == 'add':
        return global_add_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'mean':
        return global_mean_pool(x, batch_info['batch'], batch_info['num_graphs']) 
    elif method == 'max':
        return global_max_pool(x, batch_info['batch'], batch_info['num_graphs'])
    else:
        raise ValueError("Pooling method not implemented")

def kaiming_init_with_gain(x: Tensor, gain: float, a=0, mode='fan_in', nonlinearity='relu'):
    fan = _calculate_correct_fan(x, mode)
    non_linearity_gain = calculate_gain(nonlinearity, a)
    std = non_linearity_gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std * gain   # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return x.uniform_(-bound, bound)

class myLinear(nn.Module):
    """ Linear layer with potentially smaller parameters at initialization. """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, gain: float = 1.0):
        super().__init__()
        self.gain = gain
        self.lin = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        kaiming_init_with_gain(self.lin.weight, self.gain)
        if self.lin.bias is not None:
            nn.init.normal_(self.lin.bias, 0, self.gain / math.sqrt(self.lin.out_features))

    def forward(self, x):
        return self.lin.forward(x)

class UtoGlobal(nn.Module):
    def __init__(self, in_features: int , out_features: int, bias: bool, gain: float, cuda_id: int):
        super().__init__()
        # print('[UtoGlobal] cuda_id: ', cuda_id)
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.lin1 = myLinear(in_features, out_features, bias, gain=gain).to(device)
        self.lin2 = myLinear(in_features, out_features, bias, gain=gain).to(device)

    def reset_parameters(self):
        for layer in [self.lin1, self.lin2]:
            layer.reset_parameters()

    # def forward_original(self, u, batch_info: dict, method='mean'):
    #     """ u: (num_nodes, colors, in_features)
    #         output: (batch_size, out_features). """
    #     print('[UtoGlobal] input u.shape: ', u.shape)
    #     coloring = batch_info['coloring']
    #     # Extract trace
    #     index_tensor = coloring[:, :, None].expand(u.shape[0], 1, u.shape[2])
    #     extended_diag = u.gather(1, index_tensor)[:, 0, :]          # n_nodes, in_feat
    #     mean_batch_trace = pooling(extended_diag, batch_info, 'mean')    # n_graphs, in_feat
    #     out1 = self.lin1(mean_batch_trace)                   # bs, out_feat
    #     print('[UtoGlobal] extended_diag.shape:', extended_diag.shape, ' mean_batch_trace.shape:', mean_batch_trace.shape, ' out1.shape:', out1.shape)
    #     # Extract sum of elements - trace
    #     mean = torch.sum(u / batch_info['n_batch'], dim=1)  # num_nodes, in_feat
    #     batch_sum = pooling(mean, batch_info, 'mean')                    # n_graphs, in_feat
    #     batch_sum = batch_sum - mean_batch_trace                         # make the basis orthogonal
    #     out2 = self.lin2(batch_sum)  # bs, out_feat
    #     print('[UtoGlobal] mean.shape:', mean.shape, ' batch_sum.shape:', batch_sum.shape, ' out2.shape:', out2.shape)
    #     return out1 + out2
    
    def forward(self, u, batch_info: dict, method='mean'):
        """ u: (num_nodes, colors, in_features)
            output: (batch_size, out_features). """
        # print('[UtoGlobal] input u.shape: ', u.shape)
        # print('[UtoGlobal] input u.device: ', u.device, ' batch_info.device: ', batch_info['coloring'].device)
        coloring = batch_info['coloring']
        
        # Extract trace
        index_tensor = coloring[:, :, None].expand(u.shape[0], 1, u.shape[2])
        extended_diag = u.gather(1, index_tensor)[:, 0, :]          # n_nodes, in_feat  
        # mean_batch_trace = pooling(extended_diag, batch_info, 'mean')    # n_graphs, in_feat
        # out1 = self.lin1(mean_batch_trace)                   # bs, out_feat
        # print('[UtoGlobal] extended_diag.shape:', extended_diag.shape, ' mean_batch_trace.shape:', mean_batch_trace.shape, ' out1.shape:', out1.shape)
        # print('[UtoGlobal] extended_diag.device: ', extended_diag.device, ', self.lin1.device: ', next(self.lin1.parameters()).device)
        out1 = self.lin1(extended_diag)
        
        # Extract sum of elements - trace
        mean = torch.sum(u / batch_info['n_batch'], dim=1)  # num_nodes, in_feat   # compute the mean of each node
        # batch_sum = pooling(mean, batch_info, 'mean')                    # n_graphs, in_feat
        # batch_sum = batch_sum - mean_batch_trace                         # make the basis orthogonal
        # out2 = self.lin2(batch_sum)  # bs, out_feat
        # print('[UtoGlobal] mean.shape:', mean.shape, ' batch_sum.shape:', batch_sum.shape, ' out2.shape:', out2.shape)
        out2 = self.lin2(mean)
        
        
        # print('[UtoGlobal] extended_diag.shape:', extended_diag.shape, ' out1.shape:', out1.shape, ' mean.shape:', mean.shape, ' out2.shape:', out2.shape)
        return out1 + out2

class GraphExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_x: bool, simplified=False, cuda_id: int = 0):
        super().__init__()
        self.use_x, self.simplified = use_x, simplified
        # print('[GraphExtractor] cuda_id: ', cuda_id)
        self.extractor = (XtoGlobal if self.use_x else UtoGlobal)(in_features, out_features, True, 1, cuda_id) # use_x=False
        # XtoGlobal: g = global_mean_pool(x, batch_info, method="mean")
        # pooling = global_mean_pool(x, batch_info['batch'], batch_info['num_graphs']) 
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.lin = nn.Linear(out_features, out_features).to(device)

    def reset_parameters(self):
        for layer in [self.extractor, self.lin]:
            layer.reset_parameters()

    def forward(self, u: Tensor, batch_info: dict):
        # print("\n[GraphExtractor] input u:", u.shape)
        # print('[GraphExtractor] input u.device:', u.device, ' batch_info.device:', batch_info['coloring'].device)
        out = self.extractor(u, batch_info)
        # print("[GraphExtractor] after extractor out:", out.shape)
        if self.simplified:
            return out
        out = out + self.lin(F.relu(out))
        # print("[GraphExtractor] final out:", out.shape)
        return out

class EntryWiseX(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_groups=None, residual=False, cuda_id: int = 0):
        super().__init__()
        self.residual = residual
        if n_groups is None:
            n_groups = in_features
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=False).to(device)

    def forward(self, x, batch_info=None):
        """ x: N x  channels. """
        new_x = self.lin1(x.unsqueeze(-1)).squeeze()
        return (new_x + x) if self.residual else new_x

class EntrywiseU(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_towers=None, cuda_id: int = 0):
        super().__init__()
        # print('[EntrywiseU] in_features:', in_features, ' out_features:', out_features, ' num_towers:', num_towers, ' cuda_id:', cuda_id)
        if num_towers is None:
            num_towers = in_features
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=num_towers, bias=False).to(device)

    def forward(self, u):
        """ u: N x colors x channels. """
        u = u.transpose(1, 2) # transpose
        u = self.lin1(u)
        return u.transpose(1, 2)

class UtoU(nn.Module):
    def __init__(self, in_features: int, out_features: int, residual=True, n_groups=None, cuda_id: int = 0):
        super().__init__()
        if n_groups is None:
            n_groups = 1
        self.residual = residual
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True).to(device)
        self.lin2 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=False).to(device)
        self.lin3 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=False).to(device)

    def forward(self, u: Tensor, batch_info: dict = None):
        """ U: N x n_colors x channels"""
        old_u = u
        n = batch_info['num_nodes']
        num_colors = u.shape[1]
        out_feat = self.lin1.out_channels

        mask = batch_info['mask'][..., None].expand(n, num_colors, out_feat)
        normalizer = batch_info['n_batch']
        mean2 = torch.sum(u / normalizer, dim=1)     # N, in_feat
        mean2 = mean2.unsqueeze(-1)                  # N, in_feat, 1
        # 1. Transform u element-wise
        u = u.permute(0, 2, 1)                       # In conv1d, channel dimension is second
        # print('[UtoU] u.device:', u.device)
        out = self.lin1(u).permute(0, 2, 1)

        # 2. Put in self of each line the sum over each line
        # The 0.1 factor is here to bias the network in favor of learning powers of the adjacency
        z2 = self.lin2(mean2) * 0.1                       # N, out_feat, 1
        z2 = z2.transpose(1, 2)                          # N, 1, out_feat
        index_tensor = batch_info['coloring'][:, :, None].expand(out.shape[0], 1, out_feat)
        out.scatter_add_(1, index_tensor, z2)      # n, n_colors, out_feat

        # 3. Put everywhere the sum over each line
        z3 = self.lin3(mean2)                       # N, out_feat, 1
        z3 = z3.transpose(1, 2)                     # N, 1, out_feat
        out3 = z3.expand(n, num_colors, out_feat)
        out += out3 * mask * 0.1                         # Mask the extra colors
        if self.residual:
            return old_u + out
        return out

class SMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, edge_features: int, use_x: bool,
                 use_edge_features: bool, cuda_id: int = 0):
        """ Use a MLP both for the update and message function + edge features. """
        super().__init__(aggr='add', node_dim=-2 if use_x else -3)
        self.use_x, self.use_edge_features = use_x, use_edge_features
        self.in_u, self.out_u, self.edge_features = in_features, out_features, edge_features
        self.edge_nn = nn.Linear(edge_features, out_features) if use_edge_features else None

        self.message_nn = (EntryWiseX if use_x else UtoU)(in_features, out_features, # use_x = False
                                                          n_groups=num_towers, residual=False, cuda_id=cuda_id)

        args_order2 = [out_features, out_features, num_towers, cuda_id]
        entry_wise = EntryWiseX if use_x else EntrywiseU
        self.order2_i = entry_wise(*args_order2)
        self.order2_j = entry_wise(*args_order2)
        self.order2 = entry_wise(*args_order2)

        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        self.update1 = nn.Linear(2 * out_features, out_features).to(device)
        self.update2 = nn.Linear(out_features, out_features).to(device)

    def forward(self, u, edge_index, edge_attr, batch_info):
        n = batch_info['num_nodes'] # n = 
        
        u = self.message_nn(u, batch_info) 
        
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)
        
        new_u = self.propagate(edge_index, size=(n, n), u=u, u1=u1, u2=u2, edge_attr=edge_attr)
        
        new_u /= batch_info['average_edges'][:, :, 0] if self.use_x else batch_info['average_edges'] # use_x = False
        return new_u

    # Constructs messages from node j to node i
    def message(self, u_j, u1_i, u2_j, edge_attr):
        edge_feat = self.edge_nn(edge_attr) if self.use_edge_features else torch.zeros(1) # use_edge_features = False
        
        if not self.use_x: # use_x = False
            edge_feat = edge_feat.unsqueeze(1)
        
        edge_feat = edge_feat.to(u1_i.device)
        order2 = self.order2(torch.relu(u1_i + u2_j + edge_feat))
        
        u_j = u_j + order2
        
        return u_j
    
    # Updates node embeddings
    def update(self, aggr_u, u):
        up1 = self.update1(torch.cat((u, aggr_u), dim=-1))
        up2 = up1 + self.update2(up1)
        return up2 + u

class SMPModel(nn.Module):
    def __init__(self, num_input_features: int, num_edge_features: int, num_classes: int, num_layers: int,
                 hidden: int, residual: bool, use_edge_features: bool, shared_extractor: bool,
                 hidden_final: int, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool, cuda_id: int):
        """ num_input_features: number of node features
            num_edge_features: number of edge features
            num_classes: output dimension
            hidden: number of channels of the local contexts
            residual: use residual connexion after each SMP layer
            use_edge_features: if False, edge features are simply ignored
            shared extractor: share extractor among layers to reduce the number of parameters
            hidden_final: number of channels after extraction of graph features
            use_x: for ablation study, run a MPNN instead of SMP
            map_x_to_u: map the initial node features to the local context. If false, node features are ignored
            num_towers: inside each SMP layers, use towers to reduce the number of parameters
            simplified: if True, the feature extractor has less layers.
        """
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes
        self.residual = residual
        self.shared_extractor = shared_extractor

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x, cuda_id=cuda_id)
        
        self.device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu') 
        self.initial_lin = nn.Linear(num_input_features, hidden).to(self.device)

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for i in range(0, num_layers):
            self.convs.append(SMPLayer(in_features=hidden, num_towers=num_towers, out_features=hidden,
                                           edge_features=num_edge_features, use_x=use_x,
                                           use_edge_features=use_edge_features, cuda_id=cuda_id))
            self.batch_norm_list.append(BatchNorm(hidden, use_x) if i > 0 else None)

        # Feature extractors
        self.feature_extractor = GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x,
                                                    simplified=simplified, cuda_id=cuda_id)

        # Last layers
        self.after_conv = nn.Linear(hidden_final, hidden_final).to(self.device)
        self.final_lin = nn.Linear(hidden_final, num_classes).to(self.device)
        
        self.edge_index = None
        self.x = None
        self.batch = None
        self.graph_index = None
        self.edge_attr = None
        global global_cuda_id
        global_cuda_id = cuda_id
        # print('[SMPModel] global_cuda_id:', global_cuda_id)
    
    def forward(self, edge_index, node_features, batch, graph_index, edge_attr=None):
        self.edge_index, self.x, self.batch, self.graph_index, self.edge_attr = edge_index, node_features, batch, graph_index, edge_attr
        
        """ data.x: (num_nodes, num_node_features)"""
        # print('[SMPModel] input data.__dict__.keys():', data.__dict__.keys())
        # print('\n[SMPModel] input data:', data)  # DataBatch(edge_index=[2, 1154], i=[64], x=[585, 29], num_nodes=585, batch=[585], ptr=[65])
        # print('[SMPModel] input batch_size = data.i.shape =', data.i.shape)  # torch.Size([64]
        # print('\n[SMPModel] input data:', data, ', data.x.shape:', data.x.shape, ', self.edge_index.shape:', self.edge_index.shape, ', data.y.shape:', data.y.shape, ', data.batch.shape:', data.batch.shape, ', data.edge_attr.shape:', data.edge_attr.shape)
        # print('[SMPZinc] data=', data, 'data.x[0]:', data.x[0], ', data.edge_attr[0]:', data.edge_attr[0], ', data.y[0]:', data.y[0], ', data.batch[0]:', data.batch[0], ', data.ptr[0]:', data.ptr[0])
        # print('[SMPZinc] data.x.dtype:', data.x.dtype, ', data.edge_attr.dtype:', data.edge_attr.dtype, ', data.y.dtype:', data.y.dtype, ', data.batch.dtype:', data.batch.dtype, ', data.ptr.dtype:', data.ptr.dtype)
        """ 
        2. no pre_transform
        [SMPModel] input data.__dict__.keys():  dict_keys(['edge_index', 'x', 'y', 'num_nodes', 'batch'])
        [SMPModel] input data:  <lab1_task1_5_1_6.Graph object at 0x1494c98daf10> , data.x.shape: torch.Size([2998, 1]) , self.edge_index.shape: torch.Size([2, 6456]) , data.y.shape: torch.Size([128]) , data.batch.shape: torch.Size([2998]) , data.edge_attr.shape: torch.Size([6456, 1])
        """
        # [SMPModel] input data:  <lab1_task1_5_1_6.Graph object at 0x1483b440cb80> , 
        # data.x.shape: torch.Size([39, 1]) , self.edge_index.shape: torch.Size([2, 82]) , data.y.shape: torch.Size([2]) , 
        # data.batch.shape: torch.Size([39]) , data.edge_attr.shape: torch.Size([82, 1])
        if edge_attr is None:
            edge_attr = torch.ones((self.edge_index.shape[1], 1), dtype=torch.float32, device=self.x.device) # edge_attr: [num_edges, 1]
        # print('[SMPModel] edge_attr.shape:', edge_attr.shape, ', edge_attr[:3]:', edge_attr[:3])


        # Compute information about the batch
        batch_info = create_batch_info(self.edge_index, self.x, self.batch, self.graph_index, self.edge_attr, self.edge_counter)
        # print('\nbatch_info.keys():', batch_info.keys())
        # print('\nbatch_info:', batch_info)
        # print('\nbatch_info["num_nodes"]:', batch_info["num_nodes"], ', batch_info["num_graphs"]:', batch_info["num_graphs"], ', batch_info["batch"].shape:', batch_info["batch"].shape)
        # print('batch_info["n_per_graph"].shape:', batch_info['n_per_graph'].shape)
        # print('batch_info["n_batch"].shape:', batch_info['n_batch'].shape)
        # print('batch_info["average_edges"].shape:', batch_info['average_edges'].shape)
        # print('batch_info["coloring"].shape:', batch_info['coloring'].shape)
        # print('batch_info["n_colors"]:', batch_info['n_colors'])
        # print('batch_info["mask"].shape:', batch_info['mask'].shape)
        # print('batch_info["mask"].shape:', batch_info['mask'].shape, ', batch_info["mask"][0]:', batch_info['mask'][0], ', batch_info["mask"][15]:', batch_info['mask'][15], ', batch_info["mask"][16]:', batch_info['mask'][16], ', batch_info["mask"][38]:', batch_info['mask'][38])

        # Create the context matrix
        if self.use_x: # use_x=False
            assert self.x is not None
            u = self.x
        elif self.map_x_to_u: # map_x_to_u=True ✅
            u = map_x_to_u(self.x, batch_info) 
        else:
            u = self.x.new_zeros((self.x.shape[0], batch_info['n_colors']))
            u.scatter_(1, batch_info['coloring'], 1)
            u = u[..., None]
        

        out = self.no_prop(u, batch_info) 
        
        u = self.initial_lin(u) 
        
        for i in range(len(self.convs)): 
            conv = self.convs[i]
            extractor = self.feature_extractor
            bn = self.batch_norm_list[i] 
            
            if self.use_batch_norm and i > 0: # use_batch_norm = True
                u = bn(u) 
            
            u = conv(u, self.edge_index, edge_attr, batch_info) + 0
            
            global_features = extractor.forward(u, batch_info)
            
            out += global_features / len(self.convs)
        
        return out
    
    
    
    
 