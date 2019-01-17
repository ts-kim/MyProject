import argparse
import logging
import logging.handlers
import datetime
import sys
import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.models import model_1
from model.WhaleDataloader import WhaleDataloader

def evaluate(args, logger, model, data_processor, dev):
    model.eval()
    
    
    model.train()
    return

def train(args, logger):

    batch_size = args.batch_size
    num_epoch = args.num_epoch
    learning_rate = args.lr
    show_loss_term = args.show_loss_term
    device_ids = list(range(len(args.gpu.split(','))))
    processed_data_path = args.data_path
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    dt = datetime.datetime.now()
    logger.info('')
    logger.info('')
    logger.info(dt.strftime("%Y.%m.%d - %Hh:%Mm:%Ss"))
    logger.info(str(' '.join(sys.argv)))
    logger.info(' -------------------- setting --------------------')
    logger.info('')
    logger.info('    model save path : {}'.format(args.save_path))
    logger.info('    gpu : {}'.format(args.gpu))
    logger.info('    number of epochs : {}'.format(num_epoch))
    logger.info('    batch size : {}'.format(batch_size))
    logger.info('    learning rate : {}'.format(learning_rate))
    logger.info('    data path : {}'.format(processed_data_path))
    if args.load_param != 'False':
        logger.info('    loaded model parameters : {}'.format(args.load_param))
    logger.info('')
    logger.info(' -------------------- setting --------------------')

    
    transform=transforms.Compose([
        transforms.ToPILImage('RGB'),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    train_loader = WhaleDataloader(args, 'train', transform=transform)
    test_loader = WhaleDataloader(args, 'train', transform=transform)
    
    model = nn.DataParallel(model_1(),device_ids=device_ids)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if args.load_param != 'False':
        checkpoint = torch.load(args.load_param)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        epoch_range = range(epoch+1,num_epoch)
    else :
        epoch_range = range(num_epoch)    
  
 
    
    
    logger.info('')
    logger.info(' ------- model training start ------- ')

    
    for epoch in epoch_range:
        epoch_loss = 0
        step_loss = 0
        for step,batch in enumerate(train_loader):
            
            batch_X = batch['image']
            batch_y = batch['label']
            if torch.cuda.is_available() : 
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
            batch_X = model(batch_X)                
            loss= criterion(batch_X,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                   
            step_loss+=loss.item()
            epoch_loss+=loss.item()
            if step%show_loss_term == show_loss_term-1 :
                logger.debug('    epoch : {0:2d}/{1:2d}   iteration : {2:4d}/{3:4d}   loss : {4:.5f}'.format(epoch+1,num_epoch,step+1,len(train_loader),step_loss/show_loss_term))
                step_loss = 0

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, 'epoch-{0}_EM-{1:.2f}_F1-{2:.2f}'.format(epoch,EM,F1)+args.save_path)
        logger.info('    epoch {0}/{1} done | avg.loss : {2:.5f}'.format(epoch+1,num_epoch,epoch_loss/(step+1)))
    logger.info(' ------- model training completed -------')

def parse_args():
    parser = argparse.ArgumentParser('BiDAF implementation by Taesung')
    parser.add_argument('--data_path', nargs='?', type=str, default = 'dataset/', help ='dataset path')
    parser.add_argument('--gpu', nargs='?', type=str, default='0', help='device number')
    parser.add_argument('--load_param', nargs='?', type=str, default='False', help='If you want to load model parameters, write down the path of model parameters. ex) --load_param /home/taesung/bidaf/BiDAF.pth')
    parser.add_argument('--save_path', nargs='?', type=str, default='saved_model/model.pth', help='If "False", no save')
    parser.add_argument('--show_loss_term', nargs='?', type=int, default=10, help='')
    parser.add_argument('--lr',nargs='?',type=float, default=0.0003, help='learning rate')
    parser.add_argument('--num_epoch',nargs='?',type=int, default=12, help='number of epochs')
    parser.add_argument('--batch_size',nargs='?',type=int,default=32, help='batch size')
    parser.add_argument('--log',nargs='?',type=str,default='./log/new_log.log',help='log path')
    parser.add_argument('--json_path',nargs='?',type=str,default='dataset/data.json',help='json path')
    parser.add_argument('--shuffle',nargs='?',type=bool, default=True,help='train data shuffle')
    parser.add_argument('--num_workers',nargs='?',type=int,default=4, help='dataloader num_workers')
    return parser.parse_args()

def log(args):
    if os.path.isdir('/'.join(args.log.split('/')[:-1])) == False:
        os.mkdir('/'.join(args.log.split('/')[:-1]))
    if os.path.isfile(args.log) == False:
        try:
            os.utime(args.log, None)
        except OSError:
            open(args.log, 'a').close()
    logger = logging.getLogger("whale_log")
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(args.log)
    streamHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger

if __name__ == '__main__':
    args = parse_args()
    logger = log(args)
    
    if os.path.isdir('/'.join(args.save_path.split('/')[:-1])) == False:
        os.mkdir('/'.join(args.save_path.split('/')[:-1]))
    
    train(args,logger)
