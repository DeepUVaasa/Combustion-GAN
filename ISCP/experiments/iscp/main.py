import numpy as np 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from options import Options
from dataloader import *
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
import re 
opt = Options().parse()

#last_processed_data = np.load('./dataset/final_augmented_combustion_pressure_data_w20_s20_split1.npz')
#last_processed_data = np.load('./dataset/final_augmented_combustion_pressure_data_w20_s20_split2.npz')
last_processed_data = np.load('./dataset/final_augmented_combustion_pressure_data_w20_s20_split3.npz')
ftrain_data = np.asarray(last_processed_data['train_data'])
ftrain_data = np.asarray(last_processed_data['train_data'])
ftrain_data = np.asarray(last_processed_data['train_data'])
ftrain_label = np.asarray(last_processed_data['train_label'])

test_data_v = np.asarray(last_processed_data['test_data'])
test_label_v = np.asarray(last_processed_data['test_label'])

vl_label = np.asarray(last_processed_data['val_label'])
vl_data = np.asarray(last_processed_data['val_data']) 

print(ftrain_data.shape, test_data_v.shape, vl_data.shape)

scaler = MinMaxScaler() # Fit on training data
norm_ftrain_data = scaler.fit_transform(ftrain_data)  # Transform training data
norm_test_data = scaler.transform(test_data_v)  # Transform test data based on training data
norm_val_data = scaler.transform(vl_data)  # Transform test data based on training data

train_ds = VibrationDataset(data=norm_ftrain_data, label=ftrain_label,train=True)
test_ds =  VibrationDataset(data=norm_test_data, label=test_label_v,test=True)
#test_ds = VibrationDataset(data=norm_val_data, label=vl_label,test=True)
val_ds = VibrationDataset(data=norm_val_data, label=vl_label,test=True)


dataloader= {"train":torch.utils.data.DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=opt.workers,drop_last=True,pin_memory=True),
            "val":torch.utils.data.DataLoader(val_ds, batch_size=opt.batchsize, shuffle=True,num_workers=opt.workers,drop_last=True,pin_memory=True),
            "test": torch.utils.data.DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=opt.workers,drop_last=False,pin_memory=True)} #load_data(opt)

opt.data_scaler = scaler
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
