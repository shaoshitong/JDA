from EmotionModel import EmotionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
import torch.optim as optim
import torch
from dataset import get_data_deap,get_data_seed
from config import get_config

def train(Model, source_loader, target_loader):
    correct = 0
    start_time = time.time()
    for (source_data, source_label),(target_data, target_label) in zip(source_loader,target_loader):
        optimizer.zero_grad()
        data_s, label_s = source_data[0].cuda(), source_label.cuda()
        data_t, label_t = target_data.cuda(), target_label.cuda()
        data=torch.cat([data_s,data_t],0)
        label=torch.cat([label_s,label_t],0)
        Model.fit_one_step(data,label,True)
        loss=Model.pred_loss+Model.walker_loss+Model.visit_loss*0.6+Model.domain_loss+Model.l2_loss
        loss.backward(retain_graph=False)
        optimizer.step()
        e_output = torch.argmax(Model.target_logits, dim=1)
        correct += torch.eq(e_output,Model.target_y.argmax(dim=1)).float().sum().item()
    test_acc = round(correct / (len(source_loader.dataset)) * 100,3)
    use_time = time.time() - start_time
    min = int(use_time // 60)
    sum = int(use_time % 60)
    print(f"Train Epoch: [{epoch}/{config.EPOCH}], Acc: {test_acc}, Time: {min}:{sum}")
    return test_acc



if __name__ == '__main__':
    torch.cuda.empty_cache()
    total_log = []
    config=get_config()
    if config.DATA.DATASET == "deap":
        total = 32
    elif config.DATA.DATASET == "seed":
        total = 15
    else:
        raise NotImplementedError("not improve this dataset")
    for exp in range(1):
        exp = exp + 1
        subject_ACC = []
        for i in range(total):
            print('*' * 20+f'target subject: {i}'+'*'*20)
            index = np.arange(0, total, 1).tolist()
            del index[i]
            if config.DATA.DATASET == "deap":
                sample_path = config.DATA.DEAP_DATA_PATH
                source_loader, target_loader = get_data_deap(sample_path=sample_path,
                                                             index=index,
                                                             i=i,
                                                             num_classes=config.MODEL.DEAP.NUM_CLASSES,
                                                             batchsize=config.MODEL.DEAP.BATCH_SIZE)
                Model = EmotionModel(config.MODEL.DEAP.OUT*config.MODEL.DEAP.TIME_DIM,config.MODEL.DEAP.NUM_CLASSES).cuda()
                optimizer = optim.Adam(
                    # [{'params': Model.linear1.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear2.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear3.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear1.bias},
                    #  {'params': Model.linear2.bias},
                    #  {'params': Model.linear3.bias},]
                    Model.parameters(), lr=0.0001
                )

            elif config.DATA.DATASET == "seed":
                sample_path = config.DATA.SEED_DATA_PATH
                source_loader, target_loader = get_data_seed(sample_path=sample_path,
                                                             index=index,
                                                             i=i,
                                                             num_classes=config.MODEL.SEED.NUM_CLASSES,
                                                             batchsize=config.MODEL.SEED.BATCH_SIZE)
                Model = EmotionModel(config.MODEL.SEED.OUT * config.MODEL.SEED.TIME_DIM, config.MODEL.SEED.NUM_CLASSES).cuda()
                optimizer = optim.Adam(
                    # [{'params': Model.linear1.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear2.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear3.weight, 'weight_decay': 0.001},
                    #  {'params': Model.linear1.bias},
                    #  {'params': Model.linear2.bias},
                    #  {'params': Model.linear3.bias},]
                    Model.parameters(), lr=0.0001
                )

            else:
                raise NotImplementedError("not improve this dataset")
            total_test_acc=[]
            for epoch in range(1, config.EPOCH + 1):
                p = float(i) / epoch
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75
                Model.alpha=l
                if config.MODEL.IF_TURN_LR:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                test_acc=train(Model, source_loader, target_loader)
                total_test_acc.append(test_acc)
            subject_ACC.append(total_test_acc)
    for i in range(config.EPOCH):
        acc=np.array([subject_ACC[j][i] for j in range(total)]).mean()
        print(f"EPOCH {i} ,ACC {acc}")


