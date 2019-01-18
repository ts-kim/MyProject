import sys
import os
import json
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")





class WhaleDataset(Dataset):
    def __init__(self, args, split , transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        json_path = args.json_path
        self.json = json.loads(open(json_path).read())
        if split == 'train':
            self.data_dir = args.data_path+'train/'
        else :
            self.data_dir = args.data_path+'test/'
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.json[self.split])

    def __getitem__(self, idx):
        if self.split == 'train':
            img_name = os.path.join(self.data_dir, self.json[self.split][idx][0])
        else :
            img_name = os.path.join(self.data_dir, self.json[self.split][idx])
        image = io.imread(img_name)
        if len(image.shape)==2:
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image,3,axis=2)
            
        if self.split == 'train':
            label = self.json[self.split][idx][2] 
            label = torch.tensor(label).long()
            sample = {'image': image, 'label': label}
        else :
            sample = {'image': image, 'file_name' : self.json[self.split][idx] }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample
    
def WhaleDataloader(args,split,transform=None):
    if split == 'train':
        dataset = WhaleDataset(args,'train',transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=args.shuffle, num_workers=args.num_workers)
    else:
        dataset = WhaleDataset(args,'test',transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        
    
    return dataloader
