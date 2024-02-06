import torch
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from itertools import product
import numpy as np
import os
import os.path as osp
from torch_geometric.datasets import TUDataset
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Atan(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.atan(x)

class WL(torch.nn.Module):
    def __init__(self,  num_it):
        super().__init__()
        self.num_it = num_it
        self.conv = WLConv()

    def forward(self, x, edge_index):
        for _ in range(self.num_it):
            x = self.conv(x, edge_index)   
        return x

def wl_process(dataset_name, num_it, hashing = False):
    dataset_list, _ = dataset_retrieve(dataset_name)
    print(dataset_list.len())
    color_list = []
    dataset_copy = []
    loader_whole = DataLoader(dataset_list, batch_size=len(dataset_list))
    data_whole = next(iter(loader_whole))
    #data_whole.x = torch.tensor([[1] for j in range(data_whole.num_nodes)])
    data_whole = data_whole.to(device)
    col_whole = wl_colors(data_whole,num_it)
    ind_whole = 0
    for i, data in enumerate(dataset_list): 
        #data.x = torch.tensor([[1] for j in range(data.num_nodes)])
        dataset_copy.append(data)
        col = col_whole[ind_whole:ind_whole+data.num_nodes]
        
        if hashing:
            col = tuple(sorted(col.tolist()))
            color_list.append(hash(col))
        else:
            color_list.append(col)
        ind_whole+=data.num_nodes
    if hashing:
        ord_color_list, ind_inverse, count_color = np.unique(color_list, return_counts=True,
                                                    return_inverse=True)
        natural_list = np.array([j for j in range(len(ord_color_list))])
        color_list = natural_list[ind_inverse].tolist()
        amin, amax = min(color_list), max(color_list)
        for i, val in enumerate(color_list):
            color_list[i] = (val-amin) / (amax-amin) 
    sub_copy = []
    for i, g  in enumerate(dataset_copy):
        g.x = torch.tensor([[1] for j in range(g.num_nodes)],  dtype=torch.float)
        if hashing:
            g.color = torch.tensor([[color_list[i]]],  dtype=torch.float)
        else:
            g.color = color_list[i]
        sub_copy.append(g)
    return sub_copy, color_list

def dataset_retrieve(dataset):
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data_new', 'TU')
    if not osp.exists(path):
        os.makedirs(path)
    dataset_list = TUDataset(path, name=dataset).shuffle()
    return dataset_list, dataset_list.num_features



### insertion procedure
def insertion(dataset_list):
    dataset_ordered = []
    dataset_ordered.append(dataset_list[0])
    for ind in range(1, len(dataset_list)):
        g = dataset_list[ind]
        ratio = g.num_nodes/len(list(set(g.color.tolist())))
        i=0
        while i<len(dataset_ordered) and ratio > dataset_ordered[i].num_nodes/len(list(set(dataset_ordered[i].color.tolist()))):
            i+=1
        top, bot = dataset_ordered[:i], dataset_ordered[i:]
        dataset_ordered = top+[g]+bot
    return dataset_ordered


def to_pyg(A):
    G = nx.from_numpy_matrix(A).to_undirected()
    g = from_networkx(G)
    g.x = torch.tensor([[1] for i in range(np.shape(A)[0])], dtype=torch.float)
    return g

def wl_equiv_graphs():
    A1 = np.array([[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,1,1,0],
                    [0,1,1,0,0,1],[0,0,1,0,0,1],[0,0,0,1,1,0]])
    A2 = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],
                    [0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A1, A2

def complete_graph(n):
    A = np.ones((n,n))
    for i in range(np.shape(A)[0]):
        A[i,i] = 0
    return A

def cycle_graph(n):

    A = np.zeros((n,n))
    A[0,-1]=1
    A[-1,0]=1
    for i in range(n-1):
        A[i,i+1]=1
        A[i+1,i]=1
    return A

def triangles():
    A = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],
                 [0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A

def wl_colors(G,num_it):
    class WL(torch.nn.Module):
        def __init__(self,  num_it):
            super().__init__()
            self.num_it = num_it
            self.conv = WLConv()

        def forward(self, x, edge_index):
            for _ in range(self.num_it):
                x = self.conv(x, edge_index) 
            return x
    model = WL(num_it)
    model.eval()
    pred = model(G.x, G.edge_index)
    return pred

if __name__== '__main__':
    G1, G2 = wl_equiv_graphs()
    G3 = triangles()
    G4 = cycle_graph(4)
    G5 = complete_graph(6)
    data_list = [to_pyg(G4),to_pyg(G5)]
    loader = DataLoader(data_list, batch_size = 2)
    data = next(iter(loader))
    print(wl_colors(data))
    print(data)
    print(data.batch[:4])