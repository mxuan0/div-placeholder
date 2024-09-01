import os
import glob, pdb
import json
import pandas as pd

from utils import remove_extra_newlines_in_generated_captions
from torch.utils.data import Dataset
from PIL import Image


class DatacompDownloadedDataset(Dataset):
    def __init__(self, prefix, output_loc=None):
        self.prefix = prefix
        self.output_loc = output_loc

        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self):
        # Use glob to match the pattern and find all jpg files under matched directories
        search_pattern = os.path.join(self.prefix + '**', '**', '*.jpg')
        candidate_list = glob.glob(search_pattern, recursive=True)

        if self.output_loc is None:
            return candidate_list
        
        image_ids = []
        for path in os.listdir(self.output_loc):
            full_path = os.path.join(self.output_loc, path)
            if path.endswith("tsv"):
                try:
                    df = pd.read_csv(full_path, sep='\t', dtype={"image_id": str})
                except pd.errors.ParserError:
                    remove_extra_newlines_in_generated_captions(full_path)
                    df = pd.read_csv(full_path, sep='\t', dtype={"image_id": str})
                except KeyError:
                    print(full_path)
                    pdb.set_trace()

            elif path.endswith("parquet"):
                df = pd.read_parquet(full_path)
            print(full_path)
            try:
                image_ids += df["image_id"].tolist()
            except KeyError:
                pdb.set_trace()

        image_ids = set(image_ids)

        return [path for path in candidate_list if path.split("/")[-1][:-4] not in image_ids]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if image.size[0] == 1 or image.size[1] == 1:
            image = None

        json_path = image_path[:-3] + "json"
        
        with open(json_path, 'r') as file:
            data = json.load(file)
            return {
                "image_id": data["key"],
                "image": image,
                "caption": data["caption"].replace('\n', ' ').replace('\t', ' ')
            }
