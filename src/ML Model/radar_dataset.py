    import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RadarRepsDataset(Dataset):
    def __init__(self, csv_path, feature_cols, max_frames=30, max_objects=20):
        """
        csv_path: train_frames.csv / val_frames.csv / test_frames.csv
        feature_cols: list of features to use for each object
        max_frames: pad/truncate number of frames per rep
        max_objects: pad/truncate number of objects per frame
        """
        self.df = pd.read_csv(csv_path)
        self.feature_cols = feature_cols
        self.max_frames = max_frames
        self.max_objects = max_objects

        # Group by rep
        self.groups = list(self.df.groupby(["subject_id", "set_id", "rep_id"]))
        self.labels = [1 if g[1]["label"].iloc[0] == "good" else 0 for g in self.groups]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, rep_df = self.groups[idx]

        # Sort frames so sequence order is consistent
        rep_df = rep_df.sort_values("frame")

        # Get unique frames
        frames = []
        for frame_id, frame_df in rep_df.groupby("frame"):
            objects = frame_df[self.feature_cols].values

            # Pad or truncate objects
            if len(objects) < self.max_objects:
                pad_len = self.max_objects - len(objects)
                objects = np.concatenate([objects, np.zeros((pad_len, len(self.feature_cols)))], axis=0)
            else:
                objects = objects[:self.max_objects]

            frames.append(objects)

        # Pad or truncate frames
        if len(frames) < self.max_frames:
            pad_frames = [np.zeros((self.max_objects, len(self.feature_cols))) 
                          for _ in range(self.max_frames - len(frames))]
            frames = frames + pad_frames
        else:
            frames = frames[:self.max_frames]

        # Convert to tensor
        X = torch.tensor(np.array(frames), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return X, y
