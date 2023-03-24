import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv

class GINWithSkip(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_classes=1, num_layers=1):
        super(GINWithSkip, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            nnconv = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))
            self.convs.append(GINConv(nnconv, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        self.linears = nn.ModuleList()
        for i in range(num_layers):
            self.linears.append(nn.Linear(hidden_size, hidden_size))

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        # print('[GINWithSkip] input x.shape: ', x.shape, 'edge_index.shape: ', edge_index.shape)
        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # print('[GINWithSkip] convs layer:', i, 'after GINConv, x.shape: ', x.shape) # x.shape: [num_nodes, hidden_size]
            x = self.bns[i](x)
            if i != 0:
                x += xs[-1] # skip connection, 
            # print('[GINWithSkip] convs layer:', i, 'after BN, x.shape: ', x.shape) # x.shape: [num_nodes, hidden_size]
            x = F.relu(x) # activation
            xs.append(x) # store the feature of each layer

        for i, lin in enumerate(self.linears):
            # print('[GINWithSkip] linears layer:', i, 'before linear, xs[i+1].shape: ', xs[i+1].shape) # xs[i+1].shape: [num_nodes, hidden_size]
            xs[i+1] = lin(xs[i+1]) # linear transformation, 
            # print('[GINWithSkip] linears layer:', i, 'after linear, xs[i+1].shape: ', xs[i+1].shape) # xs[i+1].shape: [num_nodes, hidden_size]

        x = torch.cat(xs[1:], dim=-1) # concat all the features, 1: means from the second layer
        # print('[GINWithSkip] after cat, x.shape: ', x.shape) # x.shape: [num_nodes, hidden_size * num_layers]
        
        # x = self.linear(x)
        # # print('[GINWithSkip] after linear, x.shape: ', x.shape)

        """ 
            def forward(self, x, edge_index): 定义了 forward 方法，输入参数 x 是节点特征向量，edge_index 是边的索引。

            xs = [x] 用列表 xs 存储每一层的节点特征向量，初始值为输入的节点特征向量 x。

            for i, conv in enumerate(self.convs): 遍历每一层的 GINConv 模块，其中 i 是层数的索引，conv 是当前层的 GINConv 模块。

            x = conv(x, edge_index) 调用当前层的 GINConv 模块对节点特征向量 x 进行特征更新，输出新的节点特征向量 x。

            x = self.bns[i](x) 对新的节点特征向量 x 进行批量归一化。

            if i != 0: 如果不是第一层，则将上一层的节点特征向量与当前层的节点特征向量相加。

            x = F.relu(x) 对节点特征向量进行 ReLU 激活函数操作。

            xs.append(x) 将当前层的节点特征向量 x 存储到列表 xs 中。

            for i, lin in enumerate(self.linears): 遍历每一层的线性变换模块，其中 i 是层数的索引，lin 是当前层的线性变
        """

        return x


