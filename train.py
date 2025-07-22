from args import parse_train_opt
from TCDiff import TCDiff
import warnings
warnings.filterwarnings('ignore')
import os
import codecs as cs

def train(opt):
    # split file 
    split_file_pth = f"./data/dancernum_split/split_files/split_dancerNum_{opt.required_dancer_num}.txt"
    split_filenames = []
    with cs.open(split_file_pth, 'r') as f: 
        for line in f.readlines():
            split_filenames.append(line.strip())

    model = TCDiff(checkpoint_path = opt.checkpoint, learning_rate=opt.learning_rate, \
        window_size=opt.window_size, required_dancer_num = opt.required_dancer_num, split_file = split_filenames)
    if opt.mode == "train":
        model.train_loop(opt)
    elif opt.mode == "val_without_TrajModel":
        model.given_trajectory_generation_loop(opt)
    elif opt.mode == "test":
        model.test_loop(opt)
    else:
        raise ValueError(f"Invalid mode: {opt.mode}. Must be one of ['train', 'val_without_TrajModel', 'test'].")

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
