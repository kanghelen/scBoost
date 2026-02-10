import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
import tqdm
import scanpy as sc
import pandas as pd
import argparse
import os
import anndata as ad
# import random
import shutil
import time
# import warnings
from enum import Enum
# from PIL import Image
# from torchvision import transforms

import CustomDataset2 as CD
import models



parser = argparse.ArgumentParser(description='Singular Cell Classifications')
parser.add_argument('-train_data', metavar='DIR', nargs='?', default='./',
                    help='path to dataset (default: ???)')
parser.add_argument('-val_data', metavar="DIR", nargs='?', default='./',
                    help = 'path to validation data')
parser.add_argument('--class_key', help="Column of addata that has classificaiton values. " \
"Should list values by number rather than name. ")

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--arch', default = 'linear',
                     help='architecture to use, checks models.py for list')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('--step_size', default=30, type=int,
                    help='step size for scheduler',
                    dest='step_size')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--datasetType', default='raw', type=str,
                    help='DataType Type \"Custom [OBSM_Entry_name]\" to ensure it works') ## It might be good to add a flag to specify obsm name for 'Other' data type
parser.add_argument('--save_dir', metavar="DIR", default="temp", help="Where to Save the Results")
# parser.add_argument('--model_')



def main():
    args = parser.parse_args()
    global best_acc1
    # args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("Using CPU will be slow")


    
    """ LOAD DATA HERE"""
    """ TODO: MAKE THIS COMMAND LINE INPUT """
    train_data = args.train_data
    val_data  =  args.val_data
    train_addata = sc.read_h5ad(train_data)
    val_addata = sc.read_h5ad(val_data)
    print("Data Loaded")

    train_dataset = CD.prepareDataSet(train_addata, args.datasetType, args.class_key)
    val_dataset = CD.prepareDataSet(val_addata, args.datasetType, args.class_key)

    # print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    # print(val_loader)
    # print(len(val_loader))

    input_size = train_dataset[0][0].shape[1] #this is wrong extract from train laoder. oNly works with raw
    labels_size = len(train_addata.obs[args.class_key].unique())

    """"LOAD MODEL, and other stuff"""
    print("Size {} , {} ".format(input_size, labels_size))


    model_func = models.get_model_function(args.arch)
    model = model_func(input_size,labels_size)
    model.to(device)
    """ """
    print("Model Loaded")
    
    criterion = nn.CrossEntropyLoss().to(device)

    lr = args.lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                # weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    step_size=args.step_size
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    # train_addata.X.shape[1]

    
    """ DO TRAIN/VAL loop"""
    print("Start Training / Evaluation")

    save_path = os.path.join(args.save_dir, 'pretraining.pth.tar')
    save_checkpoint({
                'epoch': -1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': -1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, False, save_path,  file_path=args.save_dir,
                      # TODO : FILE SAVE LOCATION
        )


    if args.evaluate:
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args, device)
        print("VAL LOSS: ", val_loss, ", VAC ACC 1: ", val_acc1, ", VAL ACC 5:", val_acc5)
        return
    
    best_acc1=-1
    for epoch in range(args.epochs):
        train_loss, trainn_acc, train_acc_top5 = train(train_loader, model, criterion, optimizer, epoch, device, args)
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args, device)
        
        scheduler.step()
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)


        save_path = os.path.join(args.save_dir, '{}_checkpoint.pth.tar'.format(epoch))
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, save_path,  file_path=args.save_dir,
                      # TODO : FILE SAVE LOCATION
        )






""" Train Model For 1 Epoch """
def train( train_loader, model, criterion, optimizer, epoch, device, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    model.to(device)
        # Set the model to train mode
    model.train()
    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0
    running_top5_corrects=0
    # Iterate over the batches of the train loader
    
    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
    # for inputs, labels in tqdm.tqdm(train_loader, desc="Training"):

        #Measure DataLoading time
        data_time.update(time.time() - end)

        # print(labels)
        # print("labels length " + str(len(labels)))
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = torch.tensor(labels).to(device)

                # compute output
        outputs = model(inputs).squeeze(dim=1)
        loss = criterion(outputs, labels)        
        
        # Zero the optimizer gradients

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        #Update Progress  [FANCY METHOD TEST OUT LATER]
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
       
        # Update the running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += acc1.item() * inputs.size(0)
        running_top5_corrects += acc5.item() * inputs.size(0)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    # Calculate the train loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = running_corrects / len(train_loader)
    train_acc_top5 = running_top5_corrects / len(train_loader)

    return train_loss, train_acc, train_acc_top5


    
        # Set the model to evaluation mode

def validate(val_loader, model, criterion, args, device) :  


    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images).squeeze(dim=1)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)



    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) ,#+ (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)
    progress.display_summary()

    return losses.sum,top1.avg, top5.avg



    # device = args.gpu
    # Iterate over the batches of the validation loader
    # val_acc1_li
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader, desc="validating"):


            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # Update the running loss and accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1,5))  
            running_loss += loss.item() * inputs.size(0)
            running_corrects += acc1.item() * inputs.size(0)
            running_top5_corrects += acc5.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

    # Calculate the validation loss and accuracy
    val_loss = running_loss / len(val_loader)
    val_acc1 = running_corrects / len(val_loader)
    val_acc5 = running_top5_corrects / len(val_loader)
    return val_loss, val_acc1, val_acc5



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', file_path=''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file_path, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs


if __name__ == '__main__':
    main()
