import torch
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from itertools import product
import numpy as np
from utils import *
import os.path as osp
from torch_geometric.datasets import TUDataset
import networkx as nx



class QM9_WL(InMemoryDataset):
    def __init__(self, root, num_it,transform=None, pre_transform=None, pre_filter=None,):
        self.num_it = num_it
        self.original = 'QM9'
        super().__init__(root, transform, pre_transform, pre_filter)  
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list, _ = wl_process(self.original,self.num_it)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

class NCI1_WL(InMemoryDataset):
    def __init__(self, root, num_it,transform=None, pre_transform=None, pre_filter=None,):
        self.num_it = num_it
        self.original = 'NCI1'
        super().__init__(root, transform, pre_transform, pre_filter)  
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list, _ = wl_process(self.original,self.num_it)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

class NCI1_WL_ord(InMemoryDataset):
    def __init__(self, root, num_it,transform=None, pre_transform=None, pre_filter=None,):
        self.num_it = num_it
        super().__init__(root, transform, pre_transform, pre_filter)  
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = NCI1_WL('./data/NCI1_WL/', self.num_it )
        dataset = insertion(data_list)
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0]) 

def dataset_gen_NCI1(batch_size, N, num_colors, num_it):
    root = './data/NCI1_WL/'  
    dataset_list = NCI1_WL(root, num_it).shuffle()
    num_features = dataset_list.num_features
    num_classes = dataset_list.num_classes
    color_list=[]
    for elem in dataset_list:
        color_list.append(elem.color)
    dataset_colors = []
    dataset_red = []
    added = 0
    counter = 0
    while added<N and counter < len(dataset_list):
        g = dataset_list[counter]
        if len(list(set(dataset_colors+g.color.tolist())))<num_colors:
            dataset_colors = list(set(dataset_colors+g.color.tolist()))
            dataset_red.append(g)
            added+=1
        counter+=1
    print(counter)
    print(added)
    train_dataset = dataset_red[len(dataset_red) // 10:]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = dataset_red[:len(dataset_red) // 10]
    test_loader = DataLoader(test_dataset, batch_size)
    return train_loader, test_loader, num_features, num_classes
