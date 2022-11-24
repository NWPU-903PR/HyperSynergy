import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA
import torch
import csv
from sklearn.metrics import r2_score
class TestbedDataset(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):
        cell = self.df[idx, 0]
        cell = self.codes['cells'].item2idx.get(int(cell))
        d1 = self.df[idx, 1]
        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #synergy scores
        syn = self.df[idx, 3]
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)

        GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regression
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx, 2]
        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug2_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def R2(y,f):
    r2=r2_score(y, f)
    return r2

def save_statistics(experiment_name, line_to_add): #save
    with open("{}.csv".format(experiment_name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line_to_add)

def load_statistics(experiment_name): #load
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict