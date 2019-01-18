#!/bin/sh
mkdir dataset
kaggle competitions download humpback-whale-identification -p dataset/
mkdir dataset/test
mkdir dataset/train
mv dataset/train.zip dataset/train/.
mv dataset/test.zip dataset/test/.
cd dataset/train/
unzip train.zip
rm train.zip
cd ../test/
unzip test.zip
rm test.zip

