
from model_rep import EMBNet
import pickle
import random
from torch.optim import lr_scheduler
import torch.nn as nn
from utils_syn import *
import data_syn

# training function at each epoch
def train( model,device,train_loader,train_loader1,optimizer,lr_scheduler):

    model.train()
    LOSS=0
    for batch_idx, data in enumerate(train_loader):
         for batch_idx1, data1 in enumerate(train_loader1):
            if batch_idx1==batch_idx:
                data=data.to(device)
                data1 = data1.to(device)
                optimizer.zero_grad()

                output = model(data, data1,regress=True)
                loss = loss_fn(output, data['y'].view(-1, 1).float().to(device))
                loss.backward()
                optimizer.step()
                if lr_scheduler.get_last_lr()[0]>0.00001:
                    lr_scheduler.step()
                LOSS=loss.item()
         #print("data loader")
    return LOSS

def predicting(model,device,loader,loader0,total_labels_p,total_preds_p):
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            for batch_idx1, data1 in enumerate(loader0):
                if batch_idx1 == batch_idx:
                    data = data.to(device)
                    data1 = data1.to(device)
                    output =model(data, data1)
                    total_preds_p = torch.cat((total_preds_p, output.cpu()), 0)
                    total_labels_p = torch.cat((total_labels_p, data['y'].float().view(-1, 1).cpu()), 0)

    return total_labels_p, total_preds_p

if __name__ == '__main__':
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)
    TRAIN_BATCH_SIZE= 500
    VAL_BATCH_SIZE =1024
    LR =0.0001
    LOG_INTERVAL = 10
    NUM_EPOCHS = 500
    # few_shot setting or zero_shot setting data
    data_path = './data/data_few_zero_rich'

    fpath = 'data/sample_features/'
    codes = pickle.load(open(fpath + 'codes_cell.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature_cell.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_feature_900.p', 'rb'))

    device=torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    EMB=EMBNet(n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropoutc=0.2,dropoutf=0.3).to(device)

    print('-------------------------------you are you, i am me---------------------------')
    sum_=0
    for name,param in EMB.named_parameters():
        mul=1
        for size_ in param.shape:
            mul *=size_
        sum_ +=mul
        print('%14s: %s' % (name, param.shape))
    print('parameters number: ',sum_)
    print('------------------------------you are you, i am me !------------------------------')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(EMB.parameters(), lr=LR)
    scheduler=lr_scheduler.StepLR(optimizer,step_size=50000,gamma=1.0)

    # few_shot setting or zero_shot setting
    model_file_name = 'pretrain_representation_few_zero_setting' + '.model'

    #save_representation_model file
    model_path = 'pretrain_representation_model/'

    seed = 1500
    tranditional_train = data_syn.tranditional_model_data(train_rate=0.9, val_rate=0.1,data_path= data_path,seed=seed)
    df_train, df_val, _ = tranditional_train.get_train_batch(augment=False)
    train_sample=df_train
    print("train_sample_shape:", train_sample.shape)
    print("val_sample_shape:", df_val.shape)

    train_len = int(len(train_sample) / TRAIN_BATCH_SIZE)
    val_len = int(len(df_val) / VAL_BATCH_SIZE)

    print('Training on {} samples...'.format(len(train_sample)))
    for epoch in range(1, NUM_EPOCHS + 1):
        index = [i for i in range(len(train_sample))]
        random.shuffle(index)
        train_sample = train_sample[index]
        aver_loss = 0.
        for iteratation in range(0, train_len):
            train_data0 = train_sample[iteratation * TRAIN_BATCH_SIZE:(iteratation + 1) * TRAIN_BATCH_SIZE]
            train_data = TestbedDataset(train_data0, drug_features, cell_features, codes)
            train_data1 = TestbedDataset1(train_data0, drug_features, cell_features, codes)
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            train_loader1 = DataLoader(train_data1, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            loss_all = train(EMB, device, train_loader, train_loader1, optimizer,scheduler)
            aver_loss += loss_all
            log_interval = 30
            if iteratation % log_interval == 0 and iteratation > 0:
                cur_loss = aver_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches |  loss {:8.5f}'.format(epoch, iteratation,
                             int(len( train_loader) / TRAIN_BATCH_SIZE),cur_loss))
                aver_loss = 0
        #val
        if epoch %10 == 0:
            print('val on {} samples...'.format(len(df_val)))
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            for iteratation_val in range(0, val_len):
                val_sample = df_val[iteratation_val * VAL_BATCH_SIZE:(iteratation_val + 1) * VAL_BATCH_SIZE]
                val_data = TestbedDataset(val_sample, drug_features, cell_features, codes)
                val_data1 = TestbedDataset1(val_sample, drug_features, cell_features, codes)
                val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
                val_loader1 = DataLoader(val_data1, batch_size=VAL_BATCH_SIZE, shuffle=False)
                total_labels, total_preds = predicting(EMB, device, val_loader, val_loader1, total_labels, total_preds)
            G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), r2_score(G, P)]
            print("ret_val=", ret)
    #save model
    torch.save(EMB.state_dict(), model_path + model_file_name)
