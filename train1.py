import os.path as osp
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv
import numpy as np
from torch_geometric import seed
from model import GNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_training(batch_size, mode):

    num_layer_list = [2,3,4,5,6]
    hidden_dims = [8,16,32,64,128]
    lr = 0.001
    epochs = 500
    datasets = ["PROTEINS", "NCI1", "PTC_MR"]
    num_reps = 10

    hidden_fixed = 32
    num_layers_fixed = 3
    if mode == 'layers':
        iter_list = [[hidden_fixed, num_layer_list[i]] for i in range(len(num_layer_list))]
    elif mode == 'hidden':
        iter_list = [[hidden_dims[i], num_layers_fixed] for i in range(len(hidden_dims))]

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data_new', 'TU')

    if not osp.exists(path):
        os.makedirs(path)
    act_list = ['atan','tanh']
    raw_data = []
    for act in act_list:
        for dataset in datasets:
            dataset_name = dataset
            dataset = TUDataset(path, name=dataset).shuffle()
            for item in iter_list:
                print('Number of layers: ',str(item[1]))
                for it in range(num_reps):
                    seed.seed_everything(10*(it+1))

                    dataset.shuffle()

                    train_dataset = dataset[len(dataset) // 10:]
                    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

                    test_dataset = dataset[:len(dataset) // 10]
                    test_loader = DataLoader(test_dataset, batch_size)

                    model = GNN(dataset.num_features, item[0], 1, item[1], act).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    def train():
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            data = data.to(device)
                            y = torch.reshape(data.y, (-1,1)).float()
                            optimizer.zero_grad()
                            pred = model(data.x, data.edge_index, data.batch)
                            loss = F.binary_cross_entropy(pred, y)
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss) * data.num_graphs
                        return total_loss / len(train_loader.dataset)


                    @torch.no_grad()
                    def test(loader):
                        model.eval()
                        tot_loss = 0
                        total_correct = 0
                        for data in loader:
                            data = data.to(device)
                            pred = model(data.x, data.edge_index, data.batch)
                            pred_class = (pred>0.5).int()
                            y = torch.reshape(data.y, (-1,1))
                            loss = F.binary_cross_entropy(pred, y.float())
                            tot_loss += loss.item() * data.num_graphs
                            total_correct += int((pred_class == y).sum())
                        return tot_loss / len(loader.dataset), total_correct/len(loader.dataset)*100


                    for epoch in range(1, epochs + 1):
                        train()
                        _, train_acc = test(train_loader)
                        _, test_acc= test(test_loader)
                        diff = train_acc - test_acc
                        print(it, epoch, train_acc, test_acc, train_acc - test_acc)
                        if epoch%100==0:
                            print("Epoch {}, Diff: {}".format(epoch, diff))
                        raw_data.append({'epoch':epoch,'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc,
                                          'it': it,'layer':item[1], 'hidden': item[0]})
                
            data = pd.DataFrame.from_records(raw_data)
            data.to_csv('train1_23_11/all_data_'+dataset_name+'_'+ act + '_' + mode + '.csv')

if __name__=='__main__':
    model_training(batch_size = 32, mode= 'layers')
    model_training(batch_size = 32, mode= 'hidden')
