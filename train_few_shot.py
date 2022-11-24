import numpy as np
import cv2
import sys
import glob
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import warnings
from utils_syn import *
#from tensorboardX import SummaryWriter
import pickle

from model_few import *
import data_syn

torch.set_num_threads(int(2))
warnings.filterwarnings('ignore')

def train(train_batches, data_train, model, optimizer, optimizer_lr, lr_scheduler, device):
    best_loss = 1000
    best_r2 = 0

    total_preds_train = torch.Tensor().to(device)
    total_labels_train = torch.Tensor().to(device)
    train_loss = []

    for step in range(1, train_batches + 1):

        model.train()
        optimizer.zero_grad()
        optimizer_lr.zero_grad()
        # do training
        x_support_set, x_target = data_train.get_train_batch(augment=False)

        b, spc, t = np.shape(x_support_set)
        support_set_samples_ = x_support_set.reshape(b, spc, t)
        bt, spct, tt = np.shape(x_target)
        target_samples_ = x_target.reshape(bt, spct, tt)
        # merge support set and target set in order to share the feature extractros
        support_target_samples = np.concatenate([support_set_samples_, target_samples_], axis=1)

        b, s, t = np.shape(support_target_samples)
        support_target_samples = support_target_samples.reshape(b * s, t)
        Num_samples = len(support_target_samples)
        print("data over", Num_samples)
        train_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
        train_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

        train_loader = DataLoader(train_data, batch_size=Num_samples, shuffle=False)
        train_loader1 = DataLoader(train_data1, batch_size=Num_samples, shuffle=False)

        for batch_idx, data in enumerate(train_loader):
            for batch_idx1, data1 in enumerate(train_loader1):
                if batch_idx1 == batch_idx:
                    data = data.to(device)
                    data1 = data1.to(device)
                    all_loss, kl_div, encoder_penalty, query_loss, query_output, query_target = \
                        model.run_batch(data, data1, batch_size)
                    print('(Meta-Valid) [Step: %d/%d] KL: %4.4f Encoder Penalty: %4.4f query_loss: %4.4f' % (
                        step, train_batches, kl_div, encoder_penalty, query_loss))
                    all_loss.backward()
                    train_loss.append(all_loss.item())
                    total_preds_train = torch.cat((total_preds_train, query_output), 0)
                    total_labels_train = torch.cat((total_labels_train, query_target), 0)
                    optimizer.step()
                    optimizer_lr.step()
                    if lr_scheduler.get_last_lr()[0] > 0.0000001:
                        lr_scheduler.step()

        if step % 60 == 0:
            print(total_labels_train.size())
            print(len(train_loss))
            LOSS = sum(train_loss) / len(train_loss)
            train_total_preds = total_preds_train.cpu().detach().numpy().flatten()
            train_total_labels = total_labels_train.cpu().detach().numpy().flatten()
            total_c_spearman = spearman(train_total_labels, train_total_preds)
            total_c_loss = LOSS
            total_c_rmse = rmse(train_total_labels, train_total_preds)
            total_c_pearson = pearson(train_total_labels, train_total_preds)
            total_c_r2 = r2_score(train_total_labels, train_total_preds)

            print()
            print('=' * 50)
            print("train Epoch: {} --- Meta train Loss: {:4.4f}".format(step, total_c_loss), total_c_rmse,
                  total_c_spearman, total_c_pearson, total_c_r2)
            print('=' * 50)
            save_statistics(experiment_nameT, [step, lr_scheduler.get_last_lr()[0], total_c_rmse, total_c_spearman,
                                               total_c_pearson, total_c_r2])
            total_preds_train = torch.Tensor().to(device)
            total_labels_train = torch.Tensor().to(device)
            train_loss = []

        if step % 200 == 0:  # val model
            model.eval()
            val_losses = []
            total_preds = torch.Tensor().to(device)
            total_labels = torch.Tensor().to(device)

            for val_step in range(total_val_batches):
                optimizer.zero_grad()
                optimizer_lr.zero_grad()
                x_support_set, x_target = data_train.get_test_batch(augment=False)

                b, spc, t = np.shape(x_support_set)
                support_set_samples_ = x_support_set.reshape(b, spc, t)
                bt, spct, tt = np.shape(x_target)
                target_sample_ = x_target.reshape(bt, spct, tt)

                support_target_samples = np.concatenate([support_set_samples_, target_sample_], axis=1)

                b, s, t = np.shape(support_target_samples)
                support_target_samples = support_target_samples.reshape(b * s, t)
                Num_samples = len(support_target_samples)
                val_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
                val_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

                val_loader = DataLoader(val_data, batch_size=Num_samples, shuffle=False)
                val_loader1 = DataLoader(val_data1, batch_size=Num_samples, shuffle=False)

                # val result
                for batch_idx, data in enumerate(val_loader):
                    for batch_idx1, data1 in enumerate(val_loader1):
                        if batch_idx1 == batch_idx:
                            data = data.to(device)
                            data1 = data1.to(device)
                            val_loss, kl_div, encoder_penalty, query_loss, val_output, val_target = model.run_batch(
                                data, data1, b, False)
                            print(
                                '(Meta-Valid) [Step: %d/%d] KL: %4.4f Encoder Penalty: %4.4f query_loss: %4.4f' % (
                                    step, train_batches, kl_div, encoder_penalty, query_loss))

                            val_losses.append(val_loss.item())  #
                            total_preds = torch.cat((total_preds, val_output), 0)
                            total_labels = torch.cat((total_labels, val_target), 0)
            print(total_preds.size())
            LOSS = sum(val_losses) / len(val_losses)
            total_preds = total_preds.cpu().detach().numpy().flatten()
            total_labels = total_labels.cpu().detach().numpy().flatten()

            total_c_spearman = spearman(total_labels, total_preds)
            total_c_loss = LOSS
            total_c_rmse = rmse(total_labels, total_preds)
            total_c_pearson = pearson(total_labels, total_preds)
            total_c_r2 = r2_score(total_labels, total_preds)

            # save checkpoint
            if LOSS < best_loss or total_c_r2 > best_r2:
                best_loss = LOSS
                best_r2 = total_c_r2
                model_name = '%dk_%4.4f_model' % (step, sum(val_losses) / len(val_losses))
                # defined model name
                state = {'step': step, 'state_dict': model.state_dict()}
                if not os.path.exists('saved_model/saved_model_few_shot_setting/'):
                    os.makedirs('saved_model/saved_model_few_shot_setting/', exist_ok=False)
                save_path = "saved_model/saved_model_few_shot_setting/{}_{}.pth".format(model_name, step)
                torch.save(state, save_path)

            model.train()

            print()
            print('=' * 50)  #
            print("Validation Epoch: {} --- Meta val Loss: {:4.4f}".format(step, total_c_loss), total_c_rmse,
                  total_c_spearman, total_c_pearson, total_c_r2)
            print('=' * 50)
            print()
            print('Saving checkpoint %s...' % (experiment_nameTE))
            save_statistics(experiment_nameTE, [step, total_c_loss, total_c_rmse, total_c_spearman,
                                                total_c_pearson, total_c_r2])

if __name__ == '__main__':

    # writer = SummaryWriter('./Result_ful_syn_rest')

    fpath = 'data/sample_features/'
    codes = pickle.load(open(fpath + 'codes_cell.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature_cell.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_feature_900.p', 'rb'))

    cuda_name = 'cuda:0'

    lr = 0.001
    total_epochs = 40000
    batch_size = 50
    samples_support = 50
    samples_query = 40
    total_val_batches = 30
    embed = 128

    method = 'train 50-40 ,21 cell ,layer,dim=1,inner learning rate = 0.1,dim=128'
    device = cuda_name if torch.cuda.is_available() else "cpu"

    logs_path = 'one_shot_outputs/'
    experiment_nameT = f'cell_few_shot_train_{samples_query}qs_{samples_support}ss_{lr}'
    logs = "{}way{}shot , with {} tasks, test_batch is{},method is {} ".format(samples_support,
                                                                                        samples_query, batch_size,
                                                                                        total_val_batches,
                                                                                        method)
    save_statistics(experiment_nameT, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_nameT, ["epoch", "train_c_loss", "train_c_rmse",
                                       "train_c_spearman", "train_c_pearson", "train_c_r2"])
    experiment_nameTE = f'cell_few_shot_val_{samples_query}qs_{samples_support}ss_{lr}'
    save_statistics(experiment_nameTE, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_nameTE, ["epoch", "val_c_loss", "val_c_rmse",
                                        "val_c_spearman", "val_c_pearson", "val_c_r2"])

    save_model_name = 'cell_few_shot_' + str(samples_query) + 'qs_' + str(samples_support) + 'ss' + str(
        batch_size)

    print(device)
    mini = data_syn.MiniCellDataSet(batch_size=batch_size, samples_support=samples_support, samples_query=samples_query)
    HyperSynergy_model = HyperSynergy(num_support = samples_support, num_query=samples_query).to(device)

    model_path = './pretrain_representation_model/pretrain_representation_few_zero_setting.model'

    pretrained_dict = torch.load(model_path, map_location=device)
    # read MMN's params
    model_dict = HyperSynergy_model.extractor.state_dict()
    # read same params in two model
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # updata ,
    model_dict.update(state_dict)
    # load part of model params
    HyperSynergy_model.extractor.load_state_dict(model_dict)
    # freeze
    for p in HyperSynergy_model.extractor.parameters():

        p.requires_grad = False

    lr_list = ['inner_l_rate']   # inner learning rate
    params = [x[1] for x in list(filter(lambda kv: kv[0] not in lr_list, HyperSynergy_model.named_parameters()))]
    lr_params = [x[1] for x in list(filter(lambda kv: kv[0] in lr_list, HyperSynergy_model.named_parameters()))]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=1.0e-5)  # 1.0e-6
    # optimize inner learning rate
    optimizer_lr = torch.optim.Adam(lr_params, lr=0.1)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2500, 5000, 10000, 30000], gamma=0.5)

    # Train
    print("-------------------begin train ----------------------")
    train(total_epochs, mini, HyperSynergy_model, optimizer, optimizer_lr, scheduler, device)
