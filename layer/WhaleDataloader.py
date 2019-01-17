import sys
import os
import json
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")




class WhalesDataset(Dataset):
    def __init__(self, args, split , transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.json = json.loads(open('../dataset/data.json').read())
        if split == 'train':
            self.data_dir = args.data_path+'train/'
        else :
            self.data_dir = args.data_path+'test/'
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.json[self.split])

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir,
                                self.json[self.split][idx][0])
        image = io.imread(img_name)
        
        if self.split == 'train':
            label = self.json[self.split][idx][1] 
            sample = {'image': image, 'label': label}
        else :
            sample = {'image': image}
        
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
def WhaleDataloader(args,split,transform=None):
    if split == 'train':
        dataset = Whaledataset(args,'train',transform)
    else:
        dataset = Whaledataset(args,'test',transform)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=args.shuffle, num_workers=args.num_workers)
    return dataloader
