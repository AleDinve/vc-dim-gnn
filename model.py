from torch_geometric.nn import global_add_pool, GraphConv
import torch
from torch.nn import Linear
from utils import Atan

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')



class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers, act='atan'):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if act == 'atan':
            self.act = Atan()
        elif act == 'tanh':
            self.act = torch.nn.Tanh()
        for _ in range(num_layers):
            self.layers.append(GraphConv(input_size, hidden_size, aggr='add', bias=True))
            input_size = hidden_size
        self.linear = Linear(hidden_size, output_size).to(device)
        #self.mlp = MLP([hidden_size, 2*hidden_size, output_size]).to(device)
        
    def forward(self, x, edge_index, batch):
        for layer in self.layers:  
            x = self.act(layer(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.linear(x)
        return torch.sigmoid(x)
        
                        
