import os,pickle
import numpy as np
import torch
import torch.nn as nn

from plotUtil import plot_dist,save_pair_fig,save_plot_sample,print_network,save_plot_pair_sample,loss_plot

def tmaxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform_(mod.weight)
        #mod.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self, ngpu,opt,out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.out_z = out_z

        chans = (512, 512, 512, 512, 512, 512, 512)

        self.main = nn.Sequential(
            nn.Conv1d(opt.isize, chans[0],1, bias=False),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[0],1]),
            

            
            nn.Conv1d(chans[0],chans[1],1, bias=False),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[1],1]),

            

            nn.Conv1d(chans[1],chans[2],1, bias=False),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[2],1]),


            nn.Conv1d(chans[2],chans[3],1, bias=False),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[3],1]),


            nn.Conv1d(chans[3],chans[4],1, bias=False), 
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[4],1]),


            nn.Conv1d(chans[4],chans[5],1, bias=False),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([chans[5],1]),


            nn.Conv1d(chans[5],chans[6],1, bias=False),
            nn.LayerNorm([chans[6],1]),
            nn.LeakyReLU(0.2),

            nn.Conv1d(chans[6],self.out_z,1, bias=False)


        )
    
    def forward(self, input):     
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


    
class Decoder(nn.Module):
    def __init__(self, ngpu,opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        chans = (512, 512, 512, 512, 512, 512, 512)

        self.softplus = nn.ReLU(inplace=True) 
        self.tanh = nn.Tanh()
                
        self.dl1 = nn.Conv1d(opt.nz, chans[6],1, bias=False)

        self.dl2  = nn.Conv1d(chans[6], chans[5],1, bias=False)

        self.dl3 = nn.Conv1d(chans[5]+chans[6], chans[4],1, bias=False)

        self.dl4 = nn.Conv1d(chans[4]+chans[5], chans[3],1, bias=False)


        self.dl5 = nn.Conv1d(chans[3]+opt.nz, chans[2],1, bias=False) 


        self.dl6 = nn.Conv1d(chans[2]+chans[3], chans[1],1, bias=False)


        self.dl7 = nn.Conv1d(chans[1]+chans[2], chans[0],1, bias=False)

        self.dl8 = nn.Conv1d(chans[0]+chans[1],opt.isize,1, bias=False)

    def forward(self, input):
        dl1 = self.softplus(self.dl1(input))

        dl2 = self.softplus(self.dl2(dl1))

        joind3 = torch.cat([dl2,  dl1], 1)
        dl3 = self.softplus(self.dl3(joind3))

        joind4 = torch.cat([dl3,  dl2], 1)
        dl4 = self.softplus(self.dl4(joind4))

        joind5 = torch.cat([dl4, input], 1)
        dl5 = self.softplus(self.dl5(joind5))

        joind6 = torch.cat([dl5,  dl4], 1)
        dl6 = self.softplus(self.dl6(joind6))

        joind7 = torch.cat([dl6,  dl5], 1)
        dl7 = self.softplus(self.dl7(joind7))

        joindout = torch.cat([dl7,  dl6], 1)
        output = self.tanh(self.dl8(joindout))

        return output





    
class AD_MODEL(object):
    def __init__(self,opt,dataloader,device):
        self.G=None
        self.D=None

        self.opt=opt
        self.niter=opt.niter
        self.dataset=opt.dataset
        self.model = opt.model
        self.outf=opt.outf


    def  train(self):
        raise NotImplementedError

    def visualize_results(self, epoch,samples,is_train=True):
        if is_train:
            sub_folder="train"
        else:
            sub_folder="test"

        save_dir=os.path.join(self.outf,self.model,self.dataset,sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_sample(samples, epoch, self.dataset, num_epochs=self.niter,
                         impath=os.path.join(save_dir,'epoch%03d' % epoch + '.png'))


    def visualize_pair_results(self,epoch,samples1,samples2,is_train=True):
        if is_train:
            sub_folder="train"
        else:
            sub_folder="test"

        save_dir=os.path.join(self.outf,self.model,self.dataset,sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_pair_sample(samples1, samples2, epoch, self.dataset, num_epochs=self.niter, impath=os.path.join(save_dir,'epoch%03d' % epoch + '.png'))

    def save(self,train_hist):
        save_dir = os.path.join(self.outf, self.model, self.dataset,"model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, self.model + '_history.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)

    def save_weight_GD(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D.pkl'))

    def load(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset,"model")

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D.pkl')))


    def save_loss(self,train_hist):
        loss_plot(train_hist, os.path.join(self.outf, self.model, self.dataset), self.model)


    # def saveTestPair(self,pair,save_dir,normal_abnormal):
    #     '''

    #     :param pair: list of (input,output)
    #     :param save_dir:
    #     :return:
    #     '''
    #     assert  save_dir is not None
    #     for idx,p in enumerate(pair):
    #         input=p[0]
    #         output=p[1]
    #         save_pair_fig(input,output,os.path.join(save_dir,normal_abnormal[idx]+str(idx)+".png"))


    def saveTestPair(self,pair,save_dir):
        '''

        :param pair: list of (input,output)
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        for idx,p in enumerate(pair):
            input=p[0]
            output=p[1]
            save_pair_fig(input,output,os.path.join(save_dir,str(idx)+".png"))

    def analysisRes(self,N_res,A_res,min_score,max_score,threshold,save_dir):
        '''

        :param N_res: list of normal score
        :param A_res:  dict{ "S": list of S score, "V":...}
        :param min_score:
        :param max_score:
        :return:
        '''
        print("############   Analysis   #############")
        print("############   Threshold:{}   #############".format(threshold))
        all_abnormal_score=[]
        all_normal_score=np.array([])
        for a_type in A_res:
            a_score=A_res[a_type]
            print("*********  Type:{}  *************".format(a_type))
            normal_score=normal(N_res, min_score, max_score)
            abnormal_score=normal(a_score, min_score, max_score)
            all_abnormal_score=np.concatenate((all_abnormal_score,np.array(abnormal_score)))
            all_normal_score=normal_score
            plot_dist(normal_score,abnormal_score , str(self.opt.folder)+"_"+"N", a_type,
                      save_dir)

            TP=np.count_nonzero(abnormal_score >= threshold)
            FP=np.count_nonzero(normal_score >= threshold)
            TN=np.count_nonzero(normal_score < threshold)
            FN=np.count_nonzero(abnormal_score<threshold)
            print("TP:{}".format(TP))
            print("FP:{}".format(FP))
            print("TN:{}".format(TN))
            print("FN:{}".format(FN))
            print("Accuracy:{}".format((TP + TN) * 1.0 / (TP + TN + FP + FN)))
            print("Precision/ppv:{}".format(TP * 1.0 / (TP + FP)))
            print("sensitivity/Recall:{}".format(TP * 1.0 / (TP + FN)))
            print("specificity:{}".format(TN * 1.0 / (TN + FP)))
            print("F1:{}".format(2.0 * TP / (2 * TP + FP + FN)))

        # all_abnormal_score=np.reshape(np.array(all_abnormal_score),(-1))
        # print(all_abnormal_score.shape)
        plot_dist(all_normal_score, all_abnormal_score, str(self.opt.folder)+"_"+"N", "A",
                 save_dir)






def normal(array,min_val,max_val):
    return (array-min_val)/(max_val-min_val)
