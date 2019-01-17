import os
import pandas as pd

import json


train_csv_path = 'dataset/train.csv'
train_path = 'dataset/train/'
test_path = 'dataset/test/'
landmarks_frame = pd.read_csv(train_csv_path)


train_list = []
for i in range(len(landmarks_frame)):
    train_list.append(landmarks_frame.iloc[i].tolist())
test_list = os.listdir(test_path)

data = {}
data['train'] = train_list
data['test'] = test_list

with open('dataset/data.json', 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent="\t")
    
