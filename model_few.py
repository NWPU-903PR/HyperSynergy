
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class Weight_generstive_netwrok(nn.Module):
    def __init__(self):
        super(Weight_generstive_netwrok, self).__init__()
        self.metanet_g1 = nn.Linear(128,((512+1)*256)*1)
        self.metanet_g2 = nn.Linear(128,((256+1)*1)*1)

    def forward(self, latents):

        final1 =self.metanet_g1(latents)
        final2 =self.metanet_g2(latents)
        meta_wts_1 = final1[:, :512 * 256]
        meta_bias_1 = final1[:, 512 * 256:]
        meta_wts_2 = final2[:, :256 * 1]
        meta_bias_2 = final2[:, 256 * 1:]
        meta_wts1 = F.normalize(meta_wts_1, p=2, dim=1)
        meta_wts2 = F.normalize(meta_wts_2, p=2, dim=1)

        return  [meta_wts1,meta_bias_1,meta_wts2 ,meta_bias_2]

class Inference_model(nn.Module):
    def __init__(self):
        super(Inference_model, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.norm1 = nn.LayerNorm([64, 16, 16])
        #self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.LayerNorm([64, 8, 8])
        #self.norm2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 256, 3, 1, 1)
        self.embed_size = 512
        self.hidden_size = 128  # task embedding size
        self._cuda = True
        self.dropts=nn.Dropout2d(0.2)

    def forward(self, task_embedding):
        B,S,L=task_embedding.size()
        x=task_embedding.view(B,S,2,int((L/2)**0.5),int((L/2)**0.5))
        bc, kn, c, w, h = x.size()  # batch_size, K-shot, channel, width, high
        task_embedding= x.contiguous().view(bc * kn, c, w, h)
        task_embedding=self.dropts(task_embedding)
        x = self.pool1(F.relu(self.norm1(self.conv1(task_embedding))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.conv3(x)
        x = x.mean(dim=[2, 3])
        x = x.view(bc, kn, -1)
        x = x.mean(dim=1)
        latents,z_normal = self.reparameterize(x)
        return latents, z_normal

    def reparameterize(self,x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, :z_dim]
        c_log_var = x[:, z_dim:]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c, z_signal

class Feature_embedding_network(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropout=0.2):
        super(Feature_embedding_network, self).__init__()

        #Drugs
        self.conv1= GCNConv(num_features_xd, num_features_xd*2)
        self.conv2= GCNConv(num_features_xd*2, num_features_xd*4)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd * 2)

        # Cell lines
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)

        # Aggregation
        self.fc_g1= torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm=nn.BatchNorm1d(num_features_xd*2)

        self.fcc1 = nn.Linear(64 * 25, output_dimc)
        self.normf=nn.BatchNorm1d(2*output_dimd+output_dimc)

    def forward(self, data_train,data0_train,task_batch):
        #Drug A
        x1_train, edge_index1_train, batch1_train= data_train['x'], data_train['edge_index'] ,data_train['batch']
        x1 = self.conv1(x1_train, edge_index1_train)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1_train)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1_train)
        x1=x1+x11
        x1=self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1_train)

        #Drug B
        x2_train, edge_index2_train,batch2_train=data0_train['x'], data0_train['edge_index'],data0_train['batch']
        x2 = self.conv1(x2_train, edge_index2_train)
        x22 = self.relu(x2)
        x22 = self.conv2(x22, edge_index2_train)
        x22 = self.relu(x22)
        x22 = self.conv3(x22, edge_index2_train)
        x2=x2+x22
        x2=self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2_train)

        #Cell lines
        cell = data_train.c
        spc, h = cell.size()
        cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        cell = cell.unsqueeze(1)
        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = F.relu(self.gconv3(xt))
        xt = F.relu(self.gconv4(xt))
        bs,kn, ce,ce1 = xt.size() #batch size *(K+Q),channel, width, height
        xt = xt.view(-1,kn*ce*ce1)

        # Aggregation
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc_g2(x1)

        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2(x2)

        xt = self.relu(self.fcc1(xt))

        xc = torch.cat((x1, x2, xt), 1)
        xc = self.normf(xc)
        bskn, ce = xc.size() #batch size*(k+q), embedding size(e.g.,512)

        embeddings = xc.view(task_batch,-1,ce)

        return embeddings

class HyperSynergy(nn.Module):
    def __init__(self, num_support,num_query):

        super(HyperSynergy, self).__init__()
        self.extractor = Feature_embedding_network()
        self.inference = Inference_model()
        self.generative =Weight_generstive_netwrok()
        self.num_support = num_support  # K
        self.num_query = num_query  # Q
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([0.1]),requires_grad=True)
        #self.finetuning_lr = nn.Parameter(torch.FloatTensor([0.0001]),requires_grad=False)

    def run_batch(self, data_train,data0_train,batch_size, train1=True):

        Nq=self.num_query
        NS=self.num_support
        NB= batch_size
        kl_weight = 0.001
        encoder_penalty_weight = 1.0e-9
        Y = data_train['y'].view(NB, -1)
        support_labels= Y[:, :NS]
        target_label=Y[:,NS:]
        support_target_embeddings = self.extractor(data_train,data0_train,NB)
        support_embedings=support_target_embeddings[:,:NS]
        target_embedings = support_target_embeddings[:, NS:]
        latents, kl_div, encoder_penalty = self.meta_train_batch(support_embedings,support_labels)

        val_loss,val_output,val_target = self.inner_finetuning(
                latents,
                support_embedings,
                support_labels,
                target_embedings,
                target_label,train1)
        # calculate loss
        total_loss = val_loss + kl_weight * kl_div + encoder_penalty_weight * encoder_penalty
        return total_loss, kl_div, encoder_penalty, val_loss,val_output,val_target

    def meta_train_batch(self, inputs, target):
        train_loss1 = []
        latents,z_nomal_singal = self.inference(inputs)
        latents_init = latents
        for i in range(15):  #updata latents
            latents.retain_grad()
            specific_weights = self.generative(latents)
            train_loss,_= self.cal_target_loss(inputs, specific_weights, target, drop=0.0)
            train_loss1.append(train_loss.item())
            train_loss.backward(retain_graph=True)
            latents = latents- self.inner_l_rate * latents.grad.data # updata
        print("support set loss：",train_loss1)
        encoder_penalty = torch.mean((latents_init - latents) ** 2)
        kl_div=F.kl_div(latents.softmax(dim=-1).log(), z_nomal_singal.softmax(dim=-1), reduction='mean')
        return latents, kl_div, encoder_penalty

    def inner_finetuning(self, latents, inputs, target, val_input, val_target,train):
        specifci_weights = self.generative(latents)
        train_loss2=[]

        train_loss,_= self.cal_target_loss(inputs, specifci_weights, target,drop=0.0)
        train_loss2.append(train_loss.item())
        """
        for j in range(0):
            train_loss.backward(retain_graph=True)
            for i in range(len(specifci_weights)):
                specifci_weights[i]= specifci_weights[i]- self.finetuning_lr * specifci_weights[i].grad
                specifci_weights[i].retain_grad()
            train_loss,_ = self.cal_target_loss(inputs, specifci_weights, target,drop=0.0)
            train_loss2.append(train_loss.item())
        """
        print("w_support set loss：", train_loss2)
        val_loss,val_output = self.cal_target_loss(val_input, specifci_weights, val_target,drop=0)
        if train==False:
            train_loss.backward(retain_graph=False)
        else:
            pass
        return val_loss,val_output,val_target

    def cal_target_loss(self,input, specific_weights, target,drop):
        outputs = self.Synergy_Prediction_Network(input, specific_weights,drop)
        criterion = nn.MSELoss()
        target_loss = criterion(outputs, target)
        return target_loss,outputs

    def Synergy_Prediction_Network(self,inputs, weight,drop):
        b_size, K, embed_size = inputs.size()

        outputs=torch.Tensor().cuda()
        for i in range(b_size):
            input = inputs[i].view(K, embed_size)

            weights1 = weight[0][i].view(256, 512)
            weight_b1 = weight[1][i].view(256)
            outputs1=F.linear(input,weights1,weight_b1)
            outputs1=F.relu(outputs1)
            outputs1 = F.dropout(outputs1, drop)

            weights2=weight[2][i].view(1,256)
            weight_b2=weight[3][i].view(1)
            output=F.linear(outputs1,weights2,weight_b2)
            outputs=torch.cat([outputs,output.squeeze()],dim=0)

        return outputs.view(b_size,-1)
