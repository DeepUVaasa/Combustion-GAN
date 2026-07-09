import torch
import numpy as np 
import torch.utils.data as data
from scipy import ndimage
np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1



class VibrationDataset(data.Dataset):

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
            train_specdb = train_spec_npz
            #train_specdb = np.asarray(ndimage.median_filter(train_specdb, size=5))
            #train_specdb = normalize(train_specdb)

            train_mix = ((torch.from_numpy((train_specdb)).float())).unsqueeze(0) 

            train_mix_label = self.label[index]

            return train_mix, train_mix_label
        if self.test:
            test_spec_npz = self.npyfiles_test[index]
            
            test_specdb = test_spec_npz
            #test_specdb = np.asarray(ndimage.median_filter(test_specdb, size=5))
            #test_specdb = normalize(test_specdb)

            test_mix = ((torch.from_numpy((test_specdb)).float())).unsqueeze(0) #/255
            test_mix_label = self.labels[index]

            return test_mix,test_mix_label


    def __len__(self):
        if self.train:
            return len(self.npyfiles_train)
        if self.test:
            return len(self.npyfiles_test)
        