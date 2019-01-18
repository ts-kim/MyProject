#!/bin/sh
mkdir dataset
kaggle competitions download humpback-whale-identification -p dataset/
mkdir dataset/test
mkdir dataset/train
mv dataset/train.zip dataset/train/.
mv dataset/test.zip dataset/test/.
unzip dataset/train/train.zip
rm dataset/train/train.zip
unzip dataset/test/test.zip
rm datset/test/test.zip

