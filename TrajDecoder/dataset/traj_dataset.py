# based on group_dataset.py, 更改为仅 load traj 数据
import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v

from tqdm import tqdm

import joblib

class TrajDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "438feat",
        normalizer: Any = None,
        data_len: int = -1,
        required_dancer_num = 3,
        include_contacts: bool = True,
        force_reload: bool = False,
        split_file = None
    ):
        self.data_path = data_path
        self.raw_fps = 30
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.data_len = data_len

        # dancer_num
        self.required_dancer_num = required_dancer_num
        self.split_file = split_file

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        self.normalizer = normalizer
        # save normalizer
        if not train: 
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer_traj.pkl"), "wb")
            )
  
        # load raw data
        if (not force_reload) and os.path.exists(os.path.join(backup_path,pickle_name)): 
            print("Using cached dataset...")
        else: 
            print("Loading dataset...")
            data = self.load_aistpp(required_dancer_num)  # Call this last
    

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        # process data, convert to 6dof etc
        traj_input = self.process_dataset(data["pos"])
        self.data = {
            "traj": traj_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(traj_input) == len(data["filenames"])
        self.length = len(traj_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_)) 
        return self.data["traj"][idx], feature, filename_, self.data["wavs"][idx] 

    def load_aistpp(self, required_dancer_num = 3):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )


        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"feats438") 
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        motion_ps = sorted(glob.glob(os.path.join(motion_path, "*.pkl"))) 


        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        ignore = 0
       
        for motion_p in tqdm(motion_ps,desc='Loading'): 
            file_name = os.path.splitext(os.path.basename(motion_p))[0]
            
            file_name_origin = "_".join(os.path.splitext(os.path.basename(motion_p))[0].split("_")[:-1])
            if file_name_origin not in self.split_file: 
                continue

            if not os.path.exists(os.path.join(sound_path,file_name + '.npy')):
                ignore +=1
                continue
                
            # load motion
            data = pickle.load(open(motion_p, "rb"))
            pos = data["pos"]
            q = data["q"]

            if pos.shape[0] == required_dancer_num:
                all_pos.append(pos) 
                all_q.append(q)
                all_names.append(os.path.join(sound_path,file_name + '.npy'))
                all_wavs.append(os.path.join(wav_path,file_name + '.wav'))

        all_pos = np.array(all_pos)  
        all_q = np.array(all_q)  
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        print(f'done loading raw data, ignored {ignore}')
        return data

    def process_dataset(self, root_pos_all):
        global_pose_vec_input_all = []
        data_name = ''
        for root_pos in tqdm(root_pos_all,desc='process'):
            dancer_num, sq, c = root_pos.shape
            # to Tensor
            root_pos = torch.Tensor(root_pos)
            # to ax
            root_pos = root_pos.reshape((dancer_num, sq, -1))

            # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
            rotation = torch.Tensor(
                [0.7071068, 0.7071068, 0, 0]
            )  # 90 degrees about the x axis

            # don't forget to rotate the root position too 
            pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
            root_pos = pos_rotation.transform_points(
                root_pos
            )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

            # now, flatten everything into: batch x dancer_num x sequence x [...]
            l = [root_pos] # concat_lable, root_pos, rot6d
            global_pose_vec_input = vectorize_many(l).float().detach()

            # normalize the data. Both train and test need the same normalizer.
            if self.train:
                self.normalizer = Normalizer(global_pose_vec_input) # (bs, sq*dancer_num, 3)
            else: 
                assert self.normalizer is not None
            global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input) 

            # self.normalizer = Normalizer(global_pose_vec_input) # (bs, sq*dancer_num, 3)

            assert not torch.isnan(global_pose_vec_input).any()
            data_name = "Train" if self.train else "Test"

            # cut the dataset 
            if self.data_len > 0: # -1, not used
                global_pose_vec_input = global_pose_vec_input[: self.data_len]

            global_pose_vec_input = global_pose_vec_input.reshape(dancer_num, sq,-1) # (sq, dancer_num, 3)
            global_pose_vec_input_all.append(global_pose_vec_input)

        global_pose_vec_input_all = np.array(global_pose_vec_input_all)
        print(f"{data_name} Dataset Motion Features len is {len(global_pose_vec_input_all)}, and Dim: {global_pose_vec_input_all[0].shape}")

        return global_pose_vec_input_all


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            'feats438',
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb:
                return -1
            if aa > bb:
                return 1
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict

def cycle(iterable):
    while True:
        for x in iterable:
            yield x