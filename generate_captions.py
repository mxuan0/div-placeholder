from accessors.blip2 import Blip2Accessor
from dataloaders.url_image_loader import URLImageLoader, create_url_download_loader
from utils import read_parquet_as_df
from tqdm import tqdm

import argparse
import os
import pandas as pd


def get_captioner(captioner, device, cache_dir):
    if captioner == 'blip2':
        return Blip2Accessor(device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError


def get_image_loader(datasource, dataset_location, num_samples, batch_size):
    if datasource == 'datacomp':
        urls = read_parquet_as_df(dataset_location, num_sample=num_samples)["url"].tolist()
        return create_url_download_loader(urls, batch_size)
    else:
        raise NotImplementedError


def save_captions(ids: list, captions: list, filename):
    df = pd.DataFrame({
        'ids': ids,
        'captions': captions
    })
    df.to_csv(filename, index=False, sep='\t')
    
    ids.clear()
    captions.clear()


def main():
    parser = argparse.ArgumentParser(description="Process some keyword arguments.")

    parser.add_argument('--captioner', type=str, default='blip2')
    parser.add_argument('--cache_dir', type=str, default='/data/jingdong')
    parser.add_argument('--datasource', type=str, default='datacomp')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--dataset_location', type=str, default='/data/jingdong/datacomp/small/metadata')
    parser.add_argument('--result_location', type=str, default='/data/jingdong/datacomp/small/generated_captions/blip2-opt-6b-coco')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--storage_size', type=int, default=10000)

    args = parser.parse_args()
    
    os.makedirs(args.result_location, exist_ok=True)
    filename = f"{args.captioner}_partition"
    
    captioner = get_captioner(args.captioner, args.device, args.cache_dir)
    print("Finished loading model")

    image_loader = get_image_loader(args.datasource, args.dataset_location, args.num_samples, args.batch_size)
    print("Finished loading data loaders")

    all_captions = []
    all_ids = []
    partition = 0

    for batch_idx, batch in enumerate(tqdm(image_loader)):
        images = batch["image"]
        ids = batch["id"]

        captions = captioner.generate_caption(images)

        all_captions += captions
        all_ids += ids

        assert len(all_captions) == len(all_ids)

        if len(all_captions) > args.storage_size:
            save_captions(all_ids, all_captions, os.path.join(args.result_location, f"{filename}_{partition}.csv"))
            partition += 1
    
    if all_captions:
        save_captions(all_ids, all_captions, os.path.join(args.result_location, f"{filename}_{partition}.csv"))

if __name__ == "__main__":
    main()
