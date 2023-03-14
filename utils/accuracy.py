import torch
import numpy as np 

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) 
    iou = interArea / float(boxAArea + boxBArea - interArea )
    return iou
    
def compute_cls_acc(preds, label):
    pred = torch.max(preds, 1)[1] 
    num_correct = (pred == label).sum()
    return float(num_correct)/ float(preds.size(0))

def compute_loc_acc(loc_preds, loc_gt, theta=0.5):
    correct = 0
    for i in range(loc_preds.size(0)):
        iou = IoU(loc_preds[i], loc_gt[i])
        correct += (iou >= theta).sum()
    return float(correct) / float(loc_preds.size(0)) 

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0
    
    def updata(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt
            
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res