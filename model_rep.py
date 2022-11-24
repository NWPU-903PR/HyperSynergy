import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

# GCN based model
class EMBNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropoutc=0.2,dropoutf=0.2):

        super(EMBNet, self).__init__()
        self.n_output = n_output

        #drugs
        self.conv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv2= GCNConv(num_features_xd*2, num_features_xd*4)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd*2 )
        self.fc_g1= torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)
        self.norm=nn.BatchNorm1d(num_features_xd*2)
        self.relu = nn.ReLU()
        self.dropoutc = nn.Dropout(dropoutc)
        self.dropoutf = nn.Dropout(dropoutf)

        # cell lines (2d conv)
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.fcc1 = nn.Linear(64*25, output_dimc) #144

        # combined layers
        self.normf=nn.BatchNorm1d(2*output_dimd+output_dimc)
        self.fc1 = nn.Linear(2*output_dimd+output_dimc, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, self.n_output)

    def forward(self, data,data0,regress=True):

        x1, edge_index1, batch1= data['x'], data['edge_index'] ,data['batch']
        x2, edge_index2,batch2=data0['x'], data0['edge_index'],data0['batch']
        cell = (data['c'])

        x1 = self.conv1(x1, edge_index1)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1)

        x1=x11+x1
        x1=self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)  # global max pooling

        # flatten
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropoutc(x1)
        x1 = self.fc_g2(x1)

        x2 = self.conv1(x2, edge_index2)
        x21 = self.relu(x2)
        x21 = self.conv2(x21, edge_index2)
        x21 = self.relu(x21)

        x21 = self.conv3(x21, edge_index2)
        x2=x21+x2
        x2 = self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2) # global max pooling

        # flatten
        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropoutc(x2)
        x2 = self.fc_g2(x2)

        #cell
        spc, h = cell.size()
        cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        cell = cell.unsqueeze(1)

        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = F.relu(self.gconv3(xt))
        xt = F.relu(self.gconv4(xt))

        xt = xt.view(-1,64*5*5)
        xt=self.fcc1(xt)

        # concat
        xc = torch.cat((x1,x2, xt), 1)
        xc=self.normf(xc)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropoutf(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropoutf(xc)
        out = self.out(xc)
        return out

