import os, sys,  glob, re 
import torch
import numpy as np 
import torch.utils.data as data
import logging
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    print('getFloderK:', folder_idx, data.shape, folder_data.shape, remain_data.shape)
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y


class VibrationDataset(data.Dataset):

    def __init__(self,split,dataset_path, train=False, test=False):
        base_dir = dataset_path
        self.train = train
        self.test = test
        if self.train:
            self.npyfiles_train = self.get_instance_filenames(base_dir,split)
        if self.test:
            self.npyfiles_test, self.labels = self.get_instance_filenames(base_dir,split)

        # if (with_gt):
        #     # Used only for evaluation
        #     self.normalization_files = self.get_instance_filenames(base_dir, split, '_normalization')
        #     self.gt_files = self.get_instance_filenames(utils.concat_home_dir('/media/gpuser/seagate/LightSAL/data/dfaust/scripts'),split,'','obj')
        #     self.scans_files = self.get_instance_filenames(utils.concat_home_dir('/media/gpuser/seagate/LightSAL/data/dfaust/scans'), split,'','ply')
        #     self.shapenames = [x.split('/')[-1].split('.obj')[0] for x in self.gt_files]

    def get_instance_filenames(self,base_dir,split,ext='',format='npz'):
        npyfiles = []
        labels = []
        l = 0
        an_cnt = 0 
        n_cnt = 0 
        for dataset in split:
            for class_name in split[dataset]:
                    j = 0
                    for instanceName in split[dataset][class_name]:

                        instance_filename = os.path.join(base_dir, class_name,instanceName + "{0}.{1}".format(ext,format))
                        if not os.path.isfile(instance_filename):
                            logging.error('Requested non-existent file "' + instance_filename + "' {0} , {1}".format(l,j))
                            l = l+1
                            j = j + 1
                        npyfiles.append(instance_filename)
                        #print(instance_filename)
                        if self.test:
                            if re.search('Abnormal', instance_filename, re.IGNORECASE):
                                an_cnt = an_cnt + 1
                                labels.append(1) 
                            else:
                                n_cnt = n_cnt + 1
                                labels.append(0)
        if self.test:
            print('Number of Anomaly Sample:',an_cnt, '\n', 'Number of normal sample:',n_cnt)
            return npyfiles, labels 
        else:
            return npyfiles

    def __getitem__(self, index):
        if self.train:
            train_spec_npz = np.load(self.npyfiles_train[index])
            #train_mix = ((torch.from_numpy(resize((train_spec_npz['mix_spec'][:,:,0]), (224, 224))).float()))/255
            train_specdb = train_spec_npz['pressure']
            train_specdb = normalize(train_specdb)
            # transformer = MaxAbsScaler().fit(train_specdb)
            # train_specdb_scale = transformer.transform(train_specdb)

            train_mix = ((torch.from_numpy((train_specdb)).float())).unsqueeze(0) #/255
            # for i in range(train_mix.shape[0]):
            #     for j in range(1):
            #         train_mix[i][j] = normalize(train_mix[i][j][:])
            # train_mix = train_mix[:, :1, :]
            train_mix_label = np.zeros((train_mix.shape[0], 1))
            #sources = torch.from_numpy(train_spec_npz['sp_sources']).float()
            otherParam = torch.from_numpy(train_spec_npz['other']).float()

            return train_mix, train_mix_label, otherParam, otherParam
        if self.test:
            test_spec_npz = np.load(self.npyfiles_test[index])
            #test_mix = ((torch.from_numpy(resize((test_spec_npz['mix_spec'][:,:,0]), (224, 224))).float()))/255
            test_specdb = test_spec_npz['pressure']
            test_specdb = normalize(test_specdb)
            # transformer = MaxAbsScaler().fit(test_specdb)
            # test_specdb_scale = transformer.transform(test_specdb)
            test_mix = ((torch.from_numpy((test_specdb)).float())).unsqueeze(0) #/255
            test_mix_label = self.labels[index]
            # if re.search('anomaly', self.npyfiles_test[index], re.IGNORECASE):
            #     test_mix_label = np.ones((test_mix.shape[0], 1))
            #     #print(self.npyfiles_test[index],':::::', test_mix_label)
            # # else:
            # #     test_mix_label = np.zeros((test_mix.shape[0], 1))
            # if re.search('normal', self.npyfiles_test[index], re.IGNORECASE):
            #     test_mix_label = np.zeros((test_mix.shape[0], 1))
            #random_idx = (torch.rand(self.num_of_tr_ts_points**2) * point_set_mnlfld.shape[0]).long()
            #sources = torch.from_numpy(test_spec_npz['sp_sources']).float()
            otherParam = torch.from_numpy(test_spec_npz['other']).float()

            return test_mix,test_mix_label, otherParam, otherParam


    def __len__(self):
        if self.train:
            return len(self.npyfiles_train)
        if self.test:
            return len(self.npyfiles_test)


class VibrationDataset1(data.Dataset):

    def __init__(self,data, label, train=False, test=False):
        
        self.train = train
        self.test = test
        #print('data loader:',len(data), len(label))
        if self.train:
            self.npyfiles_train, self.label = data, label
        if self.test:
            self.npyfiles_test, self.labels = data, label

    def __getitem__(self, index):
        if self.train:
            train_spec_npz = self.npyfiles_train[index]
            #print(train_spec_npz)
            #train_mix = ((torch.from_numpy(resize((train_spec_npz['mix_spec'][:,:,0]), (224, 224))).float()))/255
            train_specdb = train_spec_npz
            #train_specdb = normalize(train_specdb)
            # transformer = MaxAbsScaler().fit(train_specdb)
            # train_specdb_scale = transformer.transform(train_specdb)

            train_mix = ((torch.from_numpy((train_specdb)).float())).unsqueeze(0) #/255
            # for i in range(train_mix.shape[0]):
            #     for j in range(1):
            #         train_mix[i][j] = normalize(train_mix[i][j][:])
            # train_mix = train_mix[:, :1, :]
            train_mix_label = self.label[index]#np.zeros((train_mix.shape[0], 1))
            #sources = torch.from_numpy(train_spec_npz['sp_sources']).float()
            otherParam = torch.from_numpy(np.array([1.0])).float()

            return train_mix, train_mix_label, otherParam, otherParam
        if self.test:
            test_spec_npz = self.npyfiles_test[index]
            #test_mix = ((torch.from_numpy(resize((test_spec_npz['mix_spec'][:,:,0]), (224, 224))).float()))/255
            test_specdb = test_spec_npz
            #test_specdb = normalize(test_specdb)

            test_mix = ((torch.from_numpy((test_specdb)).float())).unsqueeze(0) #/255
            test_mix_label = self.labels[index]

            otherParam = torch.from_numpy(np.array([1.0])).float()

            return test_mix,test_mix_label, otherParam, otherParam


    def __len__(self):
        if self.train:
            return len(self.npyfiles_train)
        if self.test:
            return len(self.npyfiles_test)
        



class SMDSegLoader(object):
    def __init__(self, data_path, mode="train"):
        self.mode = mode
        self.scaler = StandardScaler()
        print(data_path)
        data = np.load(data_path + "/SMD_train.npy")
        #self.scaler.fit(data)
        #self.scaler.fit(data)
        #data = self.scaler.transform(data)
        data = normalize(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = normalize(test_data)
        #self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.train_label = np.zeros((data_len))
        val1 = self.train[(int)(data_len * 0.97 ):] #
        val1_l = self.train_label[(int)(data_len * 0.97 ):] #
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        val2 = self.test[(int)(len(self.test) * 0.97 ):] #
        val2_l = self.test_labels[(int)(len(self.test) * 0.97 ):] #
        self.val = np.concatenate([val1, val2], 0) 
        self.val_label = np.concatenate([val1_l, val2_l], 0)
        print('data shape--> train, val, test:',self.train.shape, self.val.shape, self.val_label.shape, self.test.shape)

    def __len__(self):

        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'val'):
            return self.val.shape[0]
        elif (self.mode == 'test'):
            return self.test.shape[0] 
        else:
            return self.test.shape[0] 

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_label[index]), 0, 0
        elif (self.mode == 'val'):
            return np.float32(self.val[index]), np.float32(self.val_label[index]), 0, 0
        elif (self.mode == 'test'):
            return np.float32(self.test[index]), np.float32(self.test_labels[index]), 0, 0

def get_loader_segment(data_path, batch_size, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True
    # if mode == 'test' and mode == 'val':
    #     drp_last = True
    # else:
    #     drp_last = False 

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8, drop_last=True)
    return data_loader