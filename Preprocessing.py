import os
import pandas as pd

import json


train_csv_path = 'dataset/train.csv'
train_path = 'dataset/train/'
test_path = 'dataset/test/'
landmarks_frame = pd.read_csv(train_csv_path)

class_dict = {}
unique = landmarks_frame['Id'].unique()
for i in range(len(unique)):
    class_dict[unique[i]] = i

train_list = []
for i in range(len(landmarks_frame)):
    train_mini_list = landmarks_frame.iloc[i].tolist()
    train_mini_list += [class_dict[train_mini_list[1]]]
    train_list.append(train_mini_list)
test_list = os.listdir(test_path)

data = {}
data['train'] = train_list
data['test'] = test_list

with open('dataset/data.json', 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent="\t")
    
