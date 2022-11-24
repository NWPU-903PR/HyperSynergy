import numpy as np
import glob
import os
from tqdm import tqdm
import pandas as pd
import time
import warnings
import torch
warnings.filterwarnings('ignore')

def load_Synergydata(txt_file_dir):

    print("Loading SAMPLES...")
    def data_loader(path):

        all_files = glob.glob(os.path.join(path, "*.txt"))
        data_list = []
        for f in tqdm(all_files):
            data = pd.read_table(f,header=0, sep='\t')
            data_list.append(data)
        return data_list

    start = time.time()
    train = data_loader(os.path.join(txt_file_dir, 'train'))
    val = data_loader(os.path.join(txt_file_dir, 'val'))
    test = data_loader(os.path.join(txt_file_dir, 'test'))
    all_data = data_loader(os.path.join(txt_file_dir, 'data_rich'))
    print("Loading from raw SYNERGY data cost %.5f s" % (time.time() - start))
    return train, val, test,all_data

#for few-shot/zero-shot/lower similarity few-shot
class MiniCellDataSet():
    def __init__(self, batch_size, samples_support=10, samples_query=15, data_path= './data/data_few_zero_rich/', shuffle_cells=True):

        self.x_train, self.x_val, self.x_test, _ = load_Synergydata(data_path)
        if shuffle_cells:
            # train
            cell_ids = np.arange(len(self.x_train))
            np.random.shuffle(cell_ids)
            self.x_train1 = []
            for x in cell_ids:
                self.x_train1.append(self.x_train[x])
            self.x_train = self.x_train1
            # val
            cell_ids = np.arange(len(self.x_val))
            np.random.shuffle(cell_ids)
            self.x_val1 = []
            for x in cell_ids:
                self.x_val1.append(self.x_val[x])
            self.x_val = self.x_val1
            # test
            cell_ids = np.arange(len(self.x_test))
            np.random.shuffle(cell_ids)
            self.x_test1 = []
            for x in cell_ids:
                self.x_test1.append(self.x_test[x])
            self.x_test = self.x_test1

        self.batch_size = batch_size
        self.n_cells = 106
        self.samples_support = samples_support
        self.samples_query = samples_query
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}

    def __len__(self):
        return len(self.x_train)

    def sample_new_batch(self, data_pack):

        support_set_x = np.zeros((self.batch_size, self.samples_support, 4), dtype=np.float64)
        target_x = np.zeros((self.batch_size, self.samples_query, 4), dtype=np.float64)

        for i in range(self.batch_size):
            # Each idx in batch contains a task
            cells_idx = np.arange(len(data_pack))  #
            choose_cells = np.random.choice(cells_idx, size=1, replace=False)
            x_temp = []

            for x in choose_cells:
                samples_idx = np.arange(np.array(data_pack[x]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=self.samples_support + self.samples_query,
                                                  replace=False)
                x_temp.extend(np.array(data_pack[x])[choose_samples])
            x_temp = np.array(x_temp)
            support_set_x[i] = x_temp[:self.samples_support]
            target_x[i] = x_temp[self.samples_support: (self.samples_support + self.samples_query)]

        return torch.tensor(support_set_x), torch.tensor(target_x)

    def get_batch(self, dataset_name, augment=False):

        x_support_set, x_target = self.sample_new_batch(self.datasets[dataset_name])

        return torch.tensor(x_support_set), torch.tensor(x_target)

    def get_train_batch(self, augment=False):  # train
        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", augment)

    def get_test_batch(self, augment=False):  #test

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", augment)

    def get_val_batch(self, augment=False):  # val

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", augment)

#for data_rich and representation learning
class tranditional_model_data():
    def __init__(self, train_rate=0.64, val_rate=0.2, data_path= './data/data_few_zero_rich/', seed=1500, shuffle_cells=True):
        np.random.seed(seed)
        self.x_train, self.x_val, self.x_test, self.x_all_data = load_Synergydata(data_path)
        if shuffle_cells:
            cell_ids = np.arange(len(self.x_train))
            np.random.shuffle(cell_ids)

            self.x_train1=[]
            for x in cell_ids:
                self.x_train1.append(self.x_train[x] )
            self.x_train=self.x_train1

            cell_ids = np.arange(len(self.x_val))
            np.random.shuffle(cell_ids)
            self.x_val1 = []
            for x in cell_ids:
                self.x_val1.append(self.x_val[x])
            self.x_val=self.x_val1

            cell_ids = np.arange(len(self.x_test))
            np.random.shuffle(cell_ids)
            self.x_test1 = []
            for x in cell_ids:
                self.x_test1.append(self.x_test[x])
            self.x_test=self.x_test1

            cell_ids = np.arange(len(self.x_all_data))
            np.random.shuffle(cell_ids)
            self.x_all_data1 = []
            for x in cell_ids:
                self.x_all_data1.append(self.x_all_data[x])
            self.x_all_data = self.x_all_data1

        self.n_cell = 106
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.indexes = {"train": 0, "val": 0, "test": 0, "all_data": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test,"all_data": self.x_all_data}

    def __len__(self):
        return len(self.x_train)

    def sample_new_batch(self, data_pack):

        x_temp_train=[]
        x_temp_val = []
        x_temp_test = []

        for x in range(len(data_pack)):

                length_train =int(len(data_pack[x]) * self.train_rate)
                length_val = int(len(data_pack[x]) * self.val_rate)
                data_pack_x=np.array(data_pack[x])
                samples_idx = np.arange(data_pack_x.shape[0])
                np.random.shuffle(samples_idx)
                self.x_sample = []
                for y in samples_idx:
                    self.x_sample.append(data_pack_x[y])
                x_temp = self.x_sample
                x_temp=np.array(x_temp)

                x_temp_train.extend(x_temp[ : length_train])
                x_temp_val.extend(x_temp[length_train : (length_train+length_val)])
                x_temp_test.extend(x_temp[(length_train+length_val):])

        x_temp_train = np.array(x_temp_train)
        x_temp_val = np.array(x_temp_val)
        x_temp_test = np.array(x_temp_test)
        train_x = x_temp_train
        val_x = x_temp_val
        test_x = x_temp_test

        return train_x, val_x,test_x

    def get_batch(self, dataset_name, augment=False):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test","all_data")
        :return:
        """
        train_train,train_val,train_test = self.sample_new_batch(self.datasets[dataset_name])


        return train_train, train_val, train_test

    def get_train_batch(self, augment=False):

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", augment)

    def get_test_batch(self, augment=False):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", augment)

    def get_val_batch(self, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", augment)
    def get_all_data_batch(self, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("all_data", augment)

