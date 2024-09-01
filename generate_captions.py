from accessors.blip2 import Blip2Accessor
from accessors.llava import LlavaAccessor
from dataloaders.url_image_loader import URLImageLoader, create_captioned_image_download_loader
from datasets.datacomp_downloaded_dataset import DatacompDownloadedDataset
from torch.utils.data import DataLoader
from utils import read_parquet_as_df, filter_none_collate_fn, list_lambda, normalize_whitespace
from tqdm import tqdm
from prompts.generators.prompt_from_config import PromptFromConfig

import argparse
import os
import pdb
import torch
import pandas as pd


def main():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    parser = argparse.ArgumentParser(description="Process some keyword arguments.")

    parser.add_argument('--captioner', type=str, default='llava')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='/data/jingdong')
    parser.add_argument('--datasource', type=str, default='datacomp')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_location', type=str, default='/data/jingdong/datacomp/small/shards/00000000')
    parser.add_argument('--result_location', type=str, default='/data/jingdong/datacomp/small/generated_captions/blip2-opt-6b-coco')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--storage_size', type=int, default=10000)
    parser.add_argument('--prompt_from_config', action='store_true', default=False)

    args = parser.parse_args()
    
    os.makedirs(args.result_location, exist_ok=True)
    file_prefix = f"{args.captioner}_partition"
    
    captioner = get_captioner(args)
    captioner.model.eval()
    print("Finished loading model")

    prompts = captioner.prompts if hasattr(captioner, 'prompts') else None
    
    image_loader = get_image_loader(args)
    print("Finished loading data loaders")

    all_generated_captions = []
    all_image_ids = []
    all_original_captions = []

    partition = 0
    existing_result = os.listdir(args.result_location)
    if existing_result:
        try:
            partitions = [int(path.split('.')[0].split('_')[-1]) for path in existing_result]
            partition = max(partitions) + 1
        except Exception as e:
            print(e)
    print(f"next partition is {partition}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(image_loader)):
            images = batch["image"]

            if not images:
                continue
            
            image_ids = batch["image_id"]
            original_captions = batch['caption']

            captions_per_prompt = captioner.generate_caption(images)
                
            all_original_captions += original_captions
            all_image_ids += image_ids

            if not all_generated_captions:
                all_generated_captions = captions_per_prompt
            else:
                for i in range(len(all_generated_captions)):
                    all_generated_captions[i] += captions_per_prompt[i]

            if len(all_image_ids) > args.storage_size:
                filename = os.path.join(args.result_location, f"{file_prefix}_{partition}.parquet")
                save_captions(all_image_ids, all_original_captions, all_generated_captions, filename, prompts)
        
                partition += 1
        
        if all_original_captions:
            filename = os.path.join(args.result_location, f"{file_prefix}_{partition}.parquet")
            save_captions(all_image_ids, all_original_captions, all_generated_captions, filename, prompts)


def get_captioner(args):
    if args.captioner == 'blip2':
        return Blip2Accessor(device=args.device, cache_dir=args.cache_dir), None
    
    elif args.captioner == 'llava':
        return LlavaAccessor(args, device=args.device, cache_dir=args.cache_dir, model_type=args.model_type)
        
    else:
        raise NotImplementedError


def get_image_loader(args):
    if args.datasource == 'datacomp_url':
        df = read_parquet_as_df(args.dataset_location, num_sample=args.num_samples)
        urls = df["url"].tolist()
        original_captions = df["text"].tolist()

        return create_captioned_image_download_loader(urls, original_captions, args.batch_size, num_workers=args.num_workers)
    elif args.datasource == 'datacomp':
        return DataLoader(
            DatacompDownloadedDataset(args.dataset_location, args.result_location), 
            batch_size=args.batch_size, 
            collate_fn=filter_none_collate_fn,
            num_workers=args.num_workers
        )
    else:
        raise NotImplementedError


def save_captions(image_id: list, original_captions: list, generated_captions: list, filename: str, prompts=None):
    columns = {
        'image_id': image_id,
        'original_captions': list_lambda(original_captions, normalize_whitespace),
    }

    if prompts is None:
        caption_col_names = [f"generated_caption_group_{i}" for i in range(len(generated_captions))]
    else:
        assert len(prompts) == len(generated_captions)
        caption_col_names = prompts

    for i in range(len(caption_col_names)):
        columns[caption_col_names[i]] = list_lambda(generated_captions[i], normalize_whitespace)
    
    
    try:
        pd.DataFrame(columns).to_parquet(filename)
    except Exception:
        pdb.set_trace()
    image_id.clear()
    original_captions.clear()
    generated_captions.clear()


if __name__ == "__main__":
    main()
