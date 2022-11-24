import numpy as np
import cv2
import sys
import glob
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
import warnings
from utils_syn import *
from tensorboardX import SummaryWriter
import pickle
import random
torch.set_num_threads(int(2))
warnings.filterwarnings('ignore')
from model_few import *
import data_syn


def test(total_test_batches, data_test, model, device,best_step,best_MSE,saved_model,sup):
    # load state dict

    _load_model =saved_model +str(best_step)+'k_'+str(best_MSE)+'_model_'+str(best_step)+'.pth'
    state_dict = torch.load(_load_model)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    total_c_spearman = 0
    total_c_loss = 0
    total_c_rmse = 0
    total_c_pearson = 0
    total_c_r2 = 0
    for P in range(20):
        print("P=:", P)
        test_losses = []
        total_preds = torch.DoubleTensor().to(device)
        total_labels = torch.DoubleTensor().to(device)

        for test_step in range(total_test_batches):
            print("P:",P,"test_step:",test_step)
            x_support_set, x_target = data_test.get_test_batch(augment=False)

            b, spc, t = np.shape(x_support_set)
            support_set_images_ = x_support_set.reshape(b, spc, t)
            bt, spct, tt = np.shape(x_target)
            target_image_ = x_target.reshape(bt, spct, tt)

            support_target_images = np.concatenate([support_set_images_, target_image_], axis=1)
            b, s, t = np.shape(support_target_images)
            print(np.shape(support_target_images))
            support_target_images = support_target_images.reshape(b * s, t)
            Num_samples = len(support_target_images)
            test_data = TestbedDataset(support_target_images, drug_features, cell_features, codes)
            test_data1 = TestbedDataset1(support_target_images, drug_features, cell_features, codes)

            test_loader = DataLoader(test_data, batch_size=Num_samples, shuffle=False)
            test_loader1 = DataLoader(test_data1, batch_size=Num_samples, shuffle=False)

        # test_result
            for batch_idx, data in enumerate(test_loader):

                for batch_idx1, data1 in enumerate(test_loader1):
                    if batch_idx1 == batch_idx:

                        data = data.to(device)
                        data1 = data1.to(device)

                        _, _, _, test_loss, val_output, val_target = model.run_batch(data, data1, b,  False)
                        test_losses.append(test_loss.item())

                        total_preds = torch.cat((total_preds, val_output), 0)
                        total_labels = torch.cat((total_labels, val_target), 0)

        total_preds = total_preds.cpu().detach().numpy().flatten()
        total_labels = total_labels.cpu().detach().numpy().flatten()
        print(np.shape(total_preds))

        total_c_loss +=  mse(total_labels, total_preds)
        total_c_spearman += spearman(total_labels, total_preds)

        total_c_rmse += rmse(total_labels, total_preds)
        total_c_pearson += pearson(total_labels, total_preds)
        total_c_r2 += r2_score(total_labels, total_preds)
        save_statistics(experiment_nameTEs, [P, mse(total_labels, total_preds), rmse(total_labels, total_preds),
                                             spearman(total_labels, total_preds), pearson(total_labels, total_preds),
                                             r2_score(total_labels, total_preds)])
        np.savetxt(result_path + "label_test_" + str(sup) + "_" + str(best_MSE) + "_" + str(P) + ".txt",
         total_labels, delimiter=",")
        np.savetxt(result_path + "pred_test_" + str(sup) + "_" + str(best_MSE) + "_" + str(P) + ".txt",
        total_preds, delimiter=",")

    total_c_loss= total_c_loss / 20
    total_c_spearman=total_c_spearman / 20
    total_c_rmse=total_c_rmse / 20
    total_c_pearson=total_c_pearson / 20
    total_c_r2=total_c_r2 / 20

    print()
    print('=' * 50)

    print("Meta test Loss: {:0.05f}".format(total_c_loss), total_c_rmse,
            total_c_spearman, total_c_pearson, total_c_r2)

    save_statistics(experiment_nameTEs, ["AVE", total_c_loss, total_c_rmse, total_c_spearman,
                                         total_c_pearson, total_c_r2])
    print('=' * 50)

if __name__ == '__main__':


    cuda_name = 'cuda:0'
    device = cuda_name if torch.cuda.is_available() else "cpu"
    lr = 0.001
    batch_size = 50
    samples_support = 5 #10,30
    samples_query = 40
    total_test_batches = 200
    saved_model_path='saved_model/saved_model_few_shot_setting/'

    result_path = 'results/result_few_shot/'

    fpath = 'data/sample_features/'
    codes = pickle.load(open(fpath + 'codes_cell.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature_cell.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_feature_900.p', 'rb'))

    embed = 128
    best_MSE, best_step= format(0.0648, '.4f'), 21600  #or your trained model
    method = 'test_few_shot_cell lines, 50-5,dim=128, inner learning rate=0.1'

    experiment_nameTEs = f'cell_few_shot_test_{embed}embed_{lr}LR_{1}shot'
    logss = "{}way{}shot , with {} tasks, test_batch is{},mse is{},method is {},step is{} ".format(samples_support,
                                              samples_query,batch_size,total_test_batches,
                                               best_MSE, method,best_step)

    save_statistics(experiment_nameTEs, ["Experimental details: {}".format(logss)])
    save_statistics(experiment_nameTEs, ["epoch", "test_c_loss", "test_c_rmse",
                                         "test_c_spearman", "test_c_pearson", "test_c_r2"])
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
        # print(p)
        p.requires_grad = False

    print("-------------------begin test ----------------------")

    test(total_test_batches, mini, HyperSynergy_model, device,best_step,best_MSE,saved_model_path,samples_support)