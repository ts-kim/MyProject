
python main.py --evaluation True --resume saved_model/"$2" --batch_size 32 --version "$1" --shuffle False --log log/eval.log --gpu "$3"
