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

from model.transforms import transform_1
from model.models import model_1
from model.WhaleDataloader import WhaleDataloader

import pandas as pd

def evaluate(args, logger, model, transform):
    json_path = args.json_path
    data = json.loads(open(json_path).read())
    class_dict = data['class']
    test_loader = WhaleDataloader(args, 'test', transform=transform)
    model.eval()
    test_out = ['Image,Id']
    
    for step,batch in enumerate(test_loader):
        batch_X = batch['image']
        if args.gpu != 'False':
            batch_X = batch_X.cuda()
        with torch.no_grad():
            batch_X = model(batch_X).softmax(1)
            for i in range(len(batch_X)):
                out = batch_X[i]
                out = torch.sort(out)[1]
                out = out[:5].tolist()
                test_mini = []
                for Id, number in class_dict.items():
                    if number in out:
                        test_mini.append(Id)
                test_out.append(batch['file_name'][i]+','+' '.join(test_mini))
            
    test_out = '\n'.join(test_out)
    if os.path.isdir('submission') == False:
        os.mkdir('submission')
    
    if args.evaluation == 'False':
        with open('submission/'+args.version+'.csv','w') as f:
            f.write(test_out)
        logger.info(' ------- {} evaluation file made --------'.format(args.version) )
    else:
        with open('submission/'+args.evaluation+'.csv','w') as f:
            f.write(test_out)
        logger.info(' ------- {} evaluation file made --------'.format(args.evaluation) )
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
    logger.info('    version : {}'.format(args.version))
    logger.info('    model save path : {}'.format(args.save_path))
    logger.info('    gpu : {}'.format(args.gpu))
    logger.info('    number of epochs : {}'.format(num_epoch))
    logger.info('    batch size : {}'.format(batch_size))
    logger.info('    learning rate : {}'.format(learning_rate))
    logger.info('    data path : {}'.format(processed_data_path))
    if args.resume != 'False':
        logger.info('    loaded model parameters : {}'.format(args.resume))
    logger.info('')
    logger.info(' -------------------- setting --------------------')

    
    transform=transform_1()
    
    train_loader = WhaleDataloader(args, 'train', transform=transform)
    
    if args.gpu != 'False':
        model = nn.DataParallel(model_1(),device_ids=device_ids)
        model = model.cuda()
    else :
        model = model_1()
    
    
    if args.evaluation != 'False' :
        checkpoint = torch.load(args.evaluation)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(args, logger, model, transform)
        return
    
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if args.resume != 'False':
        checkpoint = torch.load(args.resume)
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
            if args.gpu != 'False':
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
        if os.path.isdir('saved_model') == False:
            os.mkdir('saved_model')
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, args.save_path+args.version+'_epoch-{0}'.format(epoch)+'.pth')
        logger.info('    epoch {0}/{1} done | avg.loss : {2:.5f}'.format(epoch+1,num_epoch,epoch_loss/(step+1)))
    logger.info(' ------- model training completed -------')

    evaluate(args, logger, model, transform)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser('BiDAF implementation by Taesung')
    parser.add_argument('--data_path', nargs='?', type=str, default = 'dataset/', help ='dataset path')
    parser.add_argument('--gpu', nargs='?', type=str, default='0', help='device number, cpu : False')
    parser.add_argument('--resume', nargs='?', type=str, default='False', help='If you want to load model parameters, write down the path of model parameters. ex) --resume /home/taesung/bidaf/BiDAF.pth')
    parser.add_argument('--save_path', nargs='?', type=str, default='saved_model/', help='If "False", no save')
    parser.add_argument('--show_loss_term', nargs='?', type=int, default=10, help='')
    parser.add_argument('--lr',nargs='?',type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_epoch',nargs='?',type=int, default=12, help='number of epochs')
    parser.add_argument('--batch_size',nargs='?',type=int,default=32, help='batch size')
    parser.add_argument('--log',nargs='?',type=str,default='./log/new_log.log',help='log path')
    parser.add_argument('--json_path',nargs='?',type=str,default='dataset/data.json',help='json path')
    parser.add_argument('--shuffle',nargs='?',type=bool, default=True,help='train data shuffle')
    parser.add_argument('--num_workers',nargs='?',type=int,default=4, help='dataloader num_workers')
    parser.add_argument('--evaluation',nargs='?',type=str,default='False', help='resume {input} and evaluate it (no training)')
    parser.add_argument('--version',nargs='?',type=str,default='version_default', help='version name')
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
    

    train(args,logger)
