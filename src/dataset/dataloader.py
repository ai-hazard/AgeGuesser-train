from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import numpy as np

from tqdm.auto import tqdm

class AgeDataset(Dataset):

    def __init__(self, base_path: Path, files_txt: Path, img_size: int = 224):

        self.base_path = base_path
        self.train_transforms = transforms.Compose([
            transforms.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((img_size, img_size)),
            #torchvision.transforms.RandomVerticalFlip()
        ])

        self.test_transforms = transforms.Compose([
            transforms.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((img_size, img_size)),
        ])

        self.data = []
        with open(files_txt, "r") as fp:
            lines = fp.read().split("\n")
            for line in tqdm(lines):
                line = line.strip()
                parts = line.split(",")
                if len(parts) == 2:
                    self.data.append((parts[0], int(parts[1])))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, age = self.data[idx]

        img = Image.open((self.base_path / img_name)).convert('RGB')
        x = self.train_transforms(img)
        
        return x, torch.FloatTensor(np.asarray([age]))