import os
import argparse
import torch
import torch.nn as nn 
from Model.SAT import * 
from DataLoader import *
from torch.autograd import Variable
#from asyncio import trsock
from utils.accuracy import *
from utils.lr import *
from utils.util import copy_dir, makedirs
from utils.optimizer import *
import os
import random
from skimage import measure
import cv2
from utils.func import *
from evaluator import val_loc_one_epoch 
import sys
import pprint
import shutil
from utils.optimizer import create_optimizerv2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
 
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--root', type=str, help="[CUB_200_2011, ILSVRC, OpenImage, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]", 
                                        default='ILSVRC')
parser.add_argument('--num_classes', type=int, default=200)
##  save
parser.add_argument('--save_path', type=str, default='logs')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--log_code_dir', type=str, default='save_code')
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256) 
##  dataloader
parser.add_argument('--num_workers', type=int, default=2)  
parser.add_argument('--weight_decay', type=float, default=5e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--seed', default=6, type=int) 
##  train
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')
parser.add_argument('--weight_decay_end', type=float, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--phase', type=str, default='train') 
parser.add_argument('--drop', type=float, default=0.0)  
parser.add_argument('--drop_path', type=float, default=0.1)  
parser.add_argument('--update_freq', default=1, type=int) 
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--area_thr', type=float, default=0.35)  
parser.add_argument('--lr', type=float, default=1.5e-5)
parser.add_argument('--min_lr', type=float, default=1e-6)
##  model
parser.add_argument('--arch', type=str, default='deit_sat_small_patch16_224')  
##  evaluate 
parser.add_argument('--save_img_flag', type=bool, default=False)
parser.add_argument('--save_error_flag', type=bool, default=False)
parser.add_argument('--tencrop', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)  
parser.add_argument("--local_rank", type=int,default=-1)
args = parser.parse_args() 
lr = args.lr 
#  CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch  --master_port 29501 --nproc_per_node 4 train_DDP.py
## save_log_txt
makedirs(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path + '/' + args.root + '/'  + args.arch +  '/' + args.log_code_dir + '/' + args.log_file) 
sys.stdout.log.flush()

##  save_code
save_file = ['train.py', 'evaluator.py', 'evaluator_ImageNet.py', 'train_DDP.py']
for file_name in save_file:
    shutil.copyfile(file_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['Model', 'utils', 'DataLoader']
for dir_name in save_dir:
    copy_dir(dir_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + dir_name)

## DDP
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')   
device = torch.device("cuda", local_rank)
set_seed(args.seed + dist.get_rank())

if __name__ == '__main__':
    ##  dataloader
    args.batch_size  = args.batch_size // torch.cuda.device_count()
    TrainData = eval(args.root).ImageDataset(args, phase='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(TrainData)
    Train_Loader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=args.batch_size,sampler=train_sampler,
                     num_workers=args.num_workers, pin_memory=True, drop_last=True)

    ##  model
    model = eval(args.arch)(num_classes=args.num_classes, drop_rate=args.drop, drop_path_rate=args.drop_path, pretrained=True)
    model = model.to(local_rank)
    model = DDP(model, find_unused_parameters=False , device_ids=[local_rank], output_device=local_rank)

    total_batch_size = args.batch_size * int(torch.cuda.device_count()) * args.update_freq  
    num_training_steps_per_epoch = len(TrainData) // total_batch_size
    ##  lr
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch) 
    ##  optimizer 
    optimizer = create_optimizerv2(args, model)
    loss_fnc = nn.CrossEntropyLoss()
    best_gt, best_top1, best_loc = 0, 0, 0 
    for epoch in range(0, args.epochs):
        Train_Loader.sampler.set_epoch(epoch)
        ##  accuracy
        cls_acc_1 = AverageMeter()
        loss_epoch_1 = AverageMeter()
        loss_epoch_2 = AverageMeter()
        loss_epoch_3 = AverageMeter() 
        model.train()

        for step, (path, imgs, label) in enumerate(Train_Loader):
            imgs, label = Variable(imgs).to(local_rank, non_blocking=True), label.to(local_rank, non_blocking=True)
            optimizer.zero_grad() 
            ##  loss
            area_thr = args.area_thr
            output1, ba_loss, norm_loss = model(imgs, phase='train')
            loss_cls= loss_fnc(output1, label).to(local_rank)   
            ba_loss, norm_loss = ba_loss.mean(0), norm_loss.mean(0)  
            
            loss = loss_cls +  torch.abs(ba_loss - area_thr) + norm_loss 
               
            loss.backward()
            optimizer.step()
            
            ##  count_cls_accuracy
            cur_batch = label.size(0)            
            cur_cls_acc_1 = 100. * compute_cls_acc(output1, label) 
            cls_acc_1.updata(cur_cls_acc_1, cur_batch)
            loss_epoch_1.updata(loss_cls.data, 1)
            loss_epoch_2.updata(torch.abs(ba_loss).mean(0).data, 1)
            loss_epoch_3.updata(norm_loss.data, 1)

        if dist.get_rank() == 0:
            print('Epoch:[{}/{}]\tstep:[{}/{}]\tloss_epoch_1:{:.3f}\tloss_epoch_2:{:.3f}\tloss_epoch_3:{:.3f}\tepoch_acc:{:.2f}%'.format(
                            epoch+1, args.epochs, step+1, len(Train_Loader), loss_epoch_1.avg,loss_epoch_2.avg,loss_epoch_3.avg,cls_acc_1.avg
                    ))
            sys.stdout.log.flush()
            torch.save({'model':model.state_dict(),
                        'best_thr':0,
                        'epoch':epoch+1,
                        }, os.path.join(args.save_path, args.root + '/' + args.arch + '/' + 'best_loc.pth.tar'), _use_new_zipfile_serialization=False)
            break  ## 1 epoch is enough

        