import os
import json
import torch
import einops
import random
import os.path
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from vidtransforms import *
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset


class readFeatureHMDB51(Dataset):
    def __init__(self, root: str, frames: int, fpsR: list, ensemble: int, mode: str):
        self.root = root
        self.mode = mode
        self.fpsR = fpsR
        self.frames = frames
        self.ensemble = ensemble
        self.name_map = json.load(open('../data/HMDB51/HMDB51_clsname.json'))

        if mode in ['test', 'val']:
            mode_ = 'test'
        elif mode == 'train':
            mode_ = 'train'
        split_file = os.path.join('../data/HMDB51', '{}_split01.txt'.format(mode_))
        split_info = self.read_txt(split_file)

        self.video_class_info = [self.root + vid[:-4] + 'npy' for _ in range(self.ensemble) for vid in split_info]
        self.video_path = [self.root + vid.split('/')[1][:-4] + 'npy' for _ in range(self.ensemble) for vid in split_info]

    def __len__(self):
        return len(self.video_path)

    def read_txt(self, filepath):
        file = open(filepath)
        txtdata = file.readlines()
        file.close()
        return txtdata

    def _select_indices(self, numF):
        fpsR_ = random.choice(self.fpsR)
        allInd = np.linspace(np.random.randint(1 / fpsR_), numF - 1, num=round(fpsR_ * numF), endpoint=True, dtype=int)
        if len(allInd) <= self.frames: allInd = np.linspace(0, numF - 1, num=int(numF), endpoint=True, dtype=int)
        start = np.random.randint(len(allInd) - self.frames)
        indices = allInd[start:start + self.frames]
        return indices

    def __getitem__(self, idx):
        vpath = self.video_path[idx]
        action_name_ = self.video_class_info[idx].split('/')[-2]
        action_name = self.name_map[action_name_]
        feature = torch.from_numpy(np.load(vpath))
        num_frames = feature.size(0)
        if num_frames > self.frames:
            indices = self._select_indices(num_frames)
            out = feature[indices]
        else:
            # pad by the last feature
            out = feature[-1, :][None, :].repeat([self.frames, 1])
            out[0:num_frames, :] = feature
        return out, action_name

class readFeatureHVU(Dataset):
    def __init__(self, root: str, frames: int, fpsR: list, ensemble: int, mode: str, label_file: str, annotation_file: str):
        """
        HVU Dataset Loader for multi-label classification.
        
        Args:
            root (str): Path to the directory containing video features (npy files).
            frames (int): Number of frames to sample per video.
            fpsR (list): List of FPS rates to sample from.
            ensemble (int): Number of times each video is augmented.
            mode (str): Dataset split mode ('train', 'val', 'test').
            label_file (str): Path to the label CSV file (id, label mapping).
            annotation_file (str): Path to the annotation CSV file (video paths, multi-label annotations).
        """
        self.root = root
        self.mode = mode
        self.fpsR = fpsR
        self.frames = frames
        self.ensemble = ensemble

        # Load label mapping
        self.label_map = self.load_label_map(label_file)

        # Load video annotations
        self.video_annotations = self.load_annotations(annotation_file)

        # Generate video paths for ensemble augmentation
        self.video_paths = [os.path.join(self.root, vid.split('/')[-1].replace('.mp4', '.npy')) 
                            for _ in range(self.ensemble) for vid in self.video_annotations.keys()]

    def __len__(self):
        return len(self.video_paths)

    def load_label_map(self, filepath):
        """Loads label mapping from CSV."""
        label_df = pd.read_csv(filepath)
        return {row['id']: row['label'] for _, row in label_df.iterrows()}

    def load_annotations(self, filepath):
        """Loads video annotations and converts label indices into multi-hot vectors."""
        annotation_df = pd.read_csv(filepath, header=None, names=['video', 'labels'])
        video_annotations = {}

        for _, row in annotation_df.iterrows():
            video_name = row['video']
            label_indices = list(map(int, row['labels'].split('|')))  # Convert labels to list of integers
            multi_hot_vector = np.zeros(len(self.label_map), dtype=np.float32)
            multi_hot_vector[label_indices] = 1  # Multi-hot encoding
            video_annotations[video_name] = multi_hot_vector
        
        return video_annotations

    def _select_indices(self, numF):
        """Selects frame indices using random FPS sampling."""
        fpsR_ = random.choice(self.fpsR)
        allInd = np.linspace(np.random.randint(1 / fpsR_), numF - 1, num=round(fpsR_ * numF), endpoint=True, dtype=int)
        if len(allInd) <= self.frames:
            allInd = np.linspace(0, numF - 1, num=int(numF), endpoint=True, dtype=int)
        start = np.random.randint(len(allInd) - self.frames)
        indices = allInd[start:start + self.frames]
        return indices

    def __getitem__(self, idx):
        """Loads a single video feature and its corresponding multi-label annotation."""
        vpath = self.video_paths[idx]
        video_name = os.path.basename(vpath).replace('.npy', '.mp4')

        # Load video feature
        feature = torch.from_numpy(np.load(vpath))
        num_frames = feature.size(0)

        # Sample frames
        if num_frames > self.frames:
            indices = self._select_indices(num_frames)
            out = feature[indices]
        else:
            out = feature[-1, :][None, :].repeat([self.frames, 1])  # Pad by repeating last frame
            out[0:num_frames, :] = feature

        # Get multi-hot encoded labels
        labels = torch.tensor(self.video_annotations.get(video_name, np.zeros(len(self.label_map), dtype=np.float32)))

        return out, labels

