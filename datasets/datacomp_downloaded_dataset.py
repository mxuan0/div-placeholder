import os
import glob
import json

from torch.utils.data import Dataset
from PIL import Image


class DatacompDownloadedDataset(Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self):
        # Use glob to match the pattern and find all jpg files under matched directories
        search_pattern = os.path.join(self.prefix + '**', '**', '*.jpg')
        return glob.glob(search_pattern, recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if image.size[0] == 1 or image.size[0] == 1:
            image = None

        json_path = image_path[:-3] + "json"
        
        with open(json_path, 'r') as file:
            data = json.load(file)
            return {
                "image_id": data["key"],
                "image": image,
                "caption": data["caption"].replace('\n', ' ')
            }
