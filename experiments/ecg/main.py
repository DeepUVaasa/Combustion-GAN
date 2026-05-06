import numpy as np 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from options import Options

from data import load_data

# from dcgan import DCGAN as myModel
from dataloader import *
from data_make import LoadData

device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")

opt = Options().parse()
print(opt)

# def normalize(seq):
#     '''
#     normalize to [-1,1]
#     :param seq:
#     :return:
#     '''
#     return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

# if not opt.istest:
#     if os.path.isfile('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/last_processed_data.npz'):
#         last_processed_data = np.load('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/last_processed_data.npz')
#         ftrain_data = np.asarray(last_processed_data['train_data'])
#         ftrain_label = np.asarray(last_processed_data['train_label'])
        
#         test_label_v = np.asarray(last_processed_data['test_label'])
#         test_data_v = np.asarray(last_processed_data['test_data'])
#         vl_label = np.asarray(last_processed_data['vel_label'])
#         vl_data = np.asarray(last_processed_data['vel_data']) 
#         print(ftrain_data.shape, test_data_v.shape, vl_data.shape)
#     else:      
#         train_data = np.asarray(LoadData('combustionP', 360, True))
#         random_idx = [np.random.randint(0, max(train_data.shape)) for p in range(0, max(train_data.shape))]
#         train_data = train_data[random_idx,:]
#         test_data = np.asarray(LoadData('combustionP', 360, False))
#         #print(train_data.shape, test_data.shape)
#         train_label = np.zeros(train_data.shape[0])
#         test_label = np.ones(test_data.shape[0])

#         ftrain_data = train_data[:int(0.88*train_data.shape[0]),:]
#         ftrain_label = train_label[:int(0.88*train_label.shape[0])]
#         ftest_data = np.concatenate([test_data,train_data[int(0.88*train_data.shape[0]):,:]], axis=0)
#         ftest_label = np.concatenate([test_label,train_label[int(0.88*train_label.shape[0]):]], axis=0 )
#         print(ftrain_data.shape, ftrain_label.shape, ftest_data.shape, ftest_label.shape)

#         rnd_idx = [np.random.randint(0, max(ftest_data.shape)) for p in range(0, int(0.2*max(ftest_data.shape)))]
#         #train_data, test_data = train_test_split(data, train_size = 0.98, random_state = 2021)
#         vl_data = []
#         vl_label = []

#         test_label_v = []
#         test_data_v = []

#         for ii in range(0, max(ftest_data.shape)):
#             if ii not in rnd_idx:
#                 test_data_v.append(ftest_data[ii,:])
#                 test_label_v.append(ftest_label[ii])
#             else:
#                 vl_data.append(ftest_data[ii,:])
#                 vl_label.append(ftest_label[ii])

#         test_label_v = np.asarray(test_label_v)
#         test_data_v = np.asarray(test_data_v)
#         vl_label = np.asarray(vl_label)
#         vl_data = np.asarray(vl_data)
#         print('validation data shape:',vl_data.shape, vl_label.shape, 'Test data shape:',test_data_v.shape, test_label_v.shape)

#         np.savez('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/last_processed_data.npz',train_data= ftrain_data, train_label = train_label, test_data= test_data_v, test_label=test_label_v, vel_data=vl_data, vel_label=vl_label)

# else:
    
#     if os.path.isfile('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/last_processed_data.npz'):
    
#         last_processed_data = np.load('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/last_processed_data.npz')
#         ftrain_data = np.asarray(last_processed_data['train_data'])
#         ftrain_label = np.asarray(last_processed_data['train_label'])
        
#         test_label_v = np.asarray(last_processed_data['test_label'])
#         test_data_v = np.asarray(last_processed_data['test_data'])
#         vl_label = np.asarray(last_processed_data['vel_label'])
#         vl_data = np.asarray(last_processed_data['vel_data'])
#     else:
#         train_data = np.asarray(LoadData('combustionP', 360, True))
#         random_idx = [np.random.randint(0, max(train_data.shape)) for p in range(0, max(train_data.shape))]
#         train_data = train_data[random_idx,:]
#         test_data = np.asarray(LoadData('combustionP', 360, False))
#         #print(train_data.shape, test_data.shape)
#         train_label = np.zeros(train_data.shape[0])
#         test_label = np.ones(test_data.shape[0])

#         ftrain_data = train_data[:int(0.88*train_data.shape[0]),:]
#         ftrain_label = train_label[:int(0.88*train_label.shape[0])]
#         ftest_data = np.concatenate([test_data,train_data[int(0.88*train_data.shape[0]):,:]], axis=0)
#         ftest_label = np.concatenate([test_label,train_label[int(0.88*train_label.shape[0]):]], axis=0 )
#         print(ftrain_data.shape, ftrain_label.shape, ftest_data.shape, ftest_label.shape)

#         rnd_idx = [np.random.randint(0, max(ftest_data.shape)) for p in range(0, int(0.2*max(ftest_data.shape)))]
#         #train_data, test_data = train_test_split(data, train_size = 0.98, random_state = 2021)
#         vl_data = []
#         vl_label = []

#         test_label_v = []
#         test_data_v = []

#         for ii in range(0, max(ftest_data.shape)):
#             if ii not in rnd_idx:
#                 test_data_v.append(ftest_data[ii,:])
#                 test_label_v.append(ftest_label[ii])
#             else:
#                 vl_data.append(ftest_data[ii,:])
#                 vl_label.append(ftest_label[ii])

#         test_label_v = np.asarray(test_label_v)
#         test_data_v = np.asarray(test_data_v)
#         vl_label = np.asarray(vl_label)
#         vl_data = np.asarray(vl_data)
#         ftrain_data = np.asarray(ftrain_data)
#         ftrain_label = np.asarray(ftrain_label)
#         print('validation data shape:',vl_data.shape, vl_label.shape, 'Test data shape:',test_data_v.shape, test_label_v.shape, ftrain_data.shape, ftrain_label.shape)


# print('test label:', max(test_label_v), min(test_label_v))
# syntheticData = np.load('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/synthetic_PressureData.npy')
# syntheticTestData = np.load('/media/digiaires/8TBHDD11/abol_basher_projects/DAZE/2025-twin-gpu/CombusionGAN/experiments/ecg/dataset/synth-combustionP-360-500.npy')
# syntheticTestData = np.reshape(syntheticTestData, (syntheticTestData.shape[0], syntheticTestData.shape[1]))
# print('Original data,', ftrain_data.shape, test_data_v.shape, vl_data.shape)
# print('Synthetic data shape:', syntheticData.shape, syntheticTestData.shape)

# sdata = []
# for ii in range(syntheticData.shape[0]):
#     sp = syntheticData[ii,:]
#     nsp = normalize(sp)
#     sdata.append(nsp)

# Testdata = []
# for ii in range(syntheticTestData.shape[0]):
#     Tsp = syntheticTestData[ii,:]
#     Tnsp = normalize(Tsp)
#     Testdata.append(Tnsp)

# syn_real_data = np.concatenate([ftrain_data, sdata], 0)
# #train_ds = VibrationDataset1(data=ftrain_data, label=ftrain_label,train=True)
# synlabel = ftrain_label[:syntheticData.shape[0]]
# stestl = test_label_v[:syntheticTestData.shape[0]]
# sTestLabel = np.ones_like(stestl)
# syn_real_test_label = np.concatenate([test_label_v, sTestLabel])
# syn_real_testData = np.concatenate([test_data_v,syntheticTestData])
# syn_real_label = np.concatenate([ftrain_label, synlabel],0)

# # # LPF and Normalization 
# from scipy import ndimage
# def lPF_data(data):
#     seq_list = []
#     for ii in range(0, len(data)):
#         seq = data[ii,:]
#         lfp_seq = np.asarray(ndimage.median_filter(seq, size=5))[np.newaxis, :]
#         #lfp_seq = np.asarray(normalize(seq))[np.newaxis, :]
#         seq_list.append(lfp_seq)
#     seq_arr = np.concatenate(seq_list, axis=0)
#     print('AFter LPF and Normalization:', seq_arr.shape)
#     return seq_arr

# #### Without synthetic Data #######
# # ftrain_data = lPF_data(ftrain_data)
# # test_data_v = lPF_data(test_data_v)
# # vl_data = lPF_data(vl_data)
# # train_ds = VibrationDataset1(data=ftrain_data, label=ftrain_label,train=True)
# # val_ds = VibrationDataset1(data=vl_data, label=vl_label,test=True)
# # test_ds =  VibrationDataset1(data=test_data_v, label=test_label_v,test=True)

# #### With synthetic Data #######
# # syn_real_data = lPF_data(syn_real_data)
# # syn_real_testData = lPF_data(syn_real_testData)
# # vl_data = lPF_data(vl_data)
# train_ds = VibrationDataset1(data=syn_real_data, label=syn_real_label,train=True)
# val_ds = VibrationDataset1(data=vl_data, label=vl_label,test=True)
# test_ds =  VibrationDataset1(data=syn_real_testData, label=syn_real_test_label,test=True)

# dataloader= {"train":torch.utils.data.DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=opt.workers,drop_last=True,pin_memory=True),
#             "val":torch.utils.data.DataLoader(val_ds, batch_size=opt.batchsize, shuffle=True,num_workers=opt.workers,drop_last=True,pin_memory=True),
#             "test": torch.utils.data.DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=opt.workers,drop_last=False,pin_memory=True)} #load_data(opt)


dataloader=load_data(opt)
print("load data success!!!")

if opt.model == "CombustionGAN":
    from model import CombustionGAN as MyModel

else:
    raise Exception("no this model :{}".format(opt.model))


model=MyModel(opt,dataloader,device)

if not opt.istest:
    print("################  Train  ##################")
    model.train()
else:
    print("################  Eval  ##################")
    model.load()
    model.test_type()
    # model.test_time()
    # model.plotTestFig()
    # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))
