import torch
from model import GNN
import torch
from datasets import NCI1_WL_ord
from torch_geometric import seed
import torch.nn.functional as F
from torch_geometric.nn import WLConv
from torch_geometric.loader import DataLoader
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_training(hidden_size, num_layers, it, batch_size):
    
    # train_loader, test_loader, num_features, _ = dataset_gen_NCI1(batch_size,
    #                                                                         N=1000, num_colors = 12000, num_it = num_layers)
    root = './data/NCI1_WL_ord/'  
    num_it = 4
    dataset = NCI1_WL_ord(root, num_it)
    num_features = dataset.num_features
    raw_data = []
    for data_slice in range(4):
        print(data_slice)
    #dataset = insertion(dataset_list)
        for_training = dataset[data_slice*(len(dataset) // 4):(data_slice+1)*(len(dataset) // 4)].shuffle()
        min_ratio = for_training[0].num_nodes/len(list(set(for_training[0].color.tolist())))
        max_ratio = for_training[-1].num_nodes/len(list(set(for_training[-1].color.tolist())))
        train_loader = DataLoader(for_training[len(for_training) // 10:], batch_size, shuffle=True)
        test_loader = DataLoader(for_training[:len(for_training) // 10], batch_size, shuffle=True)
        lr = 1e-3
        model = GNN(num_features, hidden_size, 1, 
                    num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = 2000
        

        @torch.enable_grad()
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
            total_correct = 0
            tot_loss = 0

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
            train_loss, train_acc = test(train_loader)
            test_loss, test_acc = test(test_loader)
            if epoch%50==0:
                print(f'epoch {epoch}')
                print(f'Train loss: {train_loss}, Train accuracy: {train_acc}%')
                print(f'Test loss: {test_loss}, Test accuracy: {test_acc}%')
                print(f'Difference of accuracy:{train_acc-test_acc}')
            raw_data.append({'Epoch': epoch, 'Train loss': train_loss, 'train accuracy':train_acc,
                            'GNN hidden dimension': hidden_size, 'seed':it,
                            'difference':train_acc-test_acc, 'dataset_slice': data_slice,
                            'max_ratio':max_ratio, 'min_ratio':min_ratio})
    
    return raw_data

if __name__ == '__main__':
    hidden_size = 16
    num_layers = 4
    batch_size = 32
    raw_data = []
    for it in range(10):
        seed.seed_everything(10*(it+1))
        raw_data += model_training(hidden_size, num_layers, it, batch_size)
    results = pd.DataFrame(raw_data)
    results.to_csv('vc_colors.csv')