import einops
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import scipy.io as sio
import h5py

class Cutout:

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, data):
        ''' img: Tensor image of size (C, H, W) '''
        _, h = data.shape
        mask = np.ones((h), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            mask[y1: y2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(data)
        return data * mask

class MyDataset(Dataset):
    def __init__(self, data, label , people_label=None,use_catout=True):
        self.data = data
        self.label = label
        self.use_catout=use_catout
        self.people_label=people_label
        if use_catout==True:
            self.catout=Cutout(n_holes=1,length=8)
        if isinstance(people_label,type(None))!=True:
            self.people_label=people_label

    def __getitem__(self, index):
        sample = self.data[index, ...]
        sample = torch.Tensor(sample)
        if self.use_catout==True:
            sample=self.catout(sample)
        label = self.label[index]
        if isinstance(self.people_label,type(None))!=True:
            people_label=self.people_label[index]
            return (sample,people_label),label
        else:
            return sample, label

    def __len__(self):
        return len(self.label)


class TrainDataset(Dataset):
    def __init__(self, data, label , people_label=None,use_catout=True,num_classes=2,training=True):
        super(Dataset,self).__init__()
        self.data = data
        self.label = label.astype(np.longlong)
        self.use_catout=use_catout
        self.people_label=people_label

        if not isinstance(self.people_label,type(None)):
            self.people_label=people_label.astype(np.longlong)
        self.num_classes=num_classes
        if num_classes==2:
            self.nums=10000
        else:
            self.nums=10000
        self.beta=0.3
        self.training=training
        self.trans=self.nums
        self.size_rate=1
        if use_catout==True:
            self.catout=Cutout(n_holes=1,length=8)
        if isinstance(people_label,type(None))!=True:
            self.people_label=people_label

    def __getitem__(self, index):
        sample = self.data[index, ...]
        sample = torch.Tensor(sample)
        label=torch.zeros(self.num_classes)
        label[self.label[index]]=1
        if self.training == True and index > 0 and index % self.nums == 0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_sample, mixup_target = self.data[mixup_idx], self.label[mixup_idx]
            mixup_label=torch.zeros(self.num_classes)
            mixup_label[self.label[mixup_idx]]=1
            beta=self.beta
            lam=np.random.beta(beta,beta)
            sample=lam*sample+(1-lam)*mixup_sample
            label=lam*label+(1-lam)*mixup_label
        if self.training==True and isinstance(self.people_label, type(None)) != True and index > 0 and index % self.nums == 0:
            people_label=torch.zeros(32)
            people_label[int(self.people_label[index])]=1
            mixup_people_label=torch.zeros(32)
            mixup_people_label[int(self.people_label[mixup_idx])] = 1
            people_label = lam * people_label + (1 - lam) * mixup_people_label
            return (sample.float(), people_label.float()), label.float()
        if self.training==True:
            p=torch.zeros(32)
            p[int(self.people_label[index])]=1
            return (sample.float(),p.float()),label.float()
        else:
            return sample.float(),label.float()

    def __len__(self):
        return len(self.label)


def get_data_deap(sample_path, index, i,num_classes, choose_index,batchsize=128):
    data=h5py.File(sample_path + "deap_data.mat")
    sample=einops.rearrange(data["feature"].__array__(float),"w h b c -> b c h w")
    elabel=einops.rearrange(data["multi_label"].__array__(np.long),"w b c -> b c w")[...,choose_index][...,None]
    print(f"the sample's shape is {sample.shape}, the elabel's shape is {elabel.shape}")
    target_sample=sample[i][None,...]
    target_elabel=elabel[i][None,...]
    source_sample=np.delete(sample,i,0) # people,film,feature
    source_elabel=np.delete(elabel,i,0)
    mean,std=torch.from_numpy(source_sample).mean([0,1,2],keepdim=True).numpy(),torch.from_numpy(source_sample).std([0,1,2],keepdim=True).numpy()
    source_sample=(source_sample-mean)/(std+1e-8)
    target_sample=(target_sample-mean)/(std+1e-8)
    people_source_elabel=np.tile(np.arange(0,sample.shape[0])[...,None],(1,sample.shape[1])).reshape(-1)
    print(people_source_elabel.shape)
    source_sample=einops.rearrange(source_sample,"b c n m -> (b c) (n m)")
    target_sample=einops.rearrange(target_sample,"b c n m -> (b c) (n m)")
    target_elabel=einops.rearrange(target_elabel,"b c n -> (b c) n").astype(np.long)
    source_elabel=einops.rearrange(source_elabel,"b c n -> (b c) n").astype(np.long)
    p = int(source_sample.shape[0] // target_sample.shape[0])
    target_sample = np.repeat(target_sample, axis=0, repeats=p)
    target_elabel = np.repeat(target_elabel, axis=0, repeats=p)
    l1,l2=target_elabel.shape[0],source_elabel.shape[0]
    print("target 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(target_elabel == 3)/l1,
          np.sum(target_elabel == 2)/l1,
          np.sum(target_elabel == 1)/l1,
          np.sum(target_elabel == 0)/l1))
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    source = TrainDataset(source_sample, source_elabel,people_label=people_source_elabel,use_catout=False,num_classes=num_classes,training=True)
    target = TrainDataset(target_sample, target_elabel,use_catout=False,num_classes=num_classes,training=False)
    source_loader = DataLoader(source, batch_size=batchsize, shuffle=True, drop_last=False)
    target_loader = DataLoader(target, batch_size=batchsize, shuffle=True, drop_last=False)
    return source_loader, target_loader

def get_data_seed(sample_path, index, i, num_classes,batchsize=128):
    target_sample = np.load(sample_path + 'person_%d data.npy' % i)
    target_elabel = np.load(sample_path + 'label.npy')
    source_sample = []
    source_elabel = []
    people_source_elabel=[]
    print("train:", index)
    for j in index:
        y=j
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'label.npy')
        source_people_elabel = np.array([y]*t_source_elabel.shape[0], dtype=float)
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
        people_source_elabel.append(source_people_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    people_source_elabel = np.concatenate(people_source_elabel, axis=0)
    p = int(source_sample.shape[0] // target_sample.shape[0])
    target_sample = np.repeat(target_sample, axis=0, repeats=p)
    target_elabel = np.repeat(target_elabel, axis=0, repeats=p)
    l1,l2=target_elabel.shape[0],source_elabel.shape[0]
    print("target 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(target_elabel == 3)/l1,
          np.sum(target_elabel == 2)/l1,
          np.sum(target_elabel == 1)/l1,
          np.sum(target_elabel == 0)/l1))
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    source = TrainDataset(source_sample, source_elabel,people_label=people_source_elabel,use_catout=False,num_classes=num_classes,training=True)
    target = TrainDataset(target_sample, target_elabel,use_catout=False,num_classes=num_classes,training=False)
    source_loader = DataLoader(source, batch_size=batchsize, shuffle=True, drop_last=False)
    target_loader = DataLoader(target, batch_size=batchsize, shuffle=True, drop_last=False)
    return source_loader, target_loader
