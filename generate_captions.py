from accessors.blip2 import Blip2Accessor
from accessors.llava import LlavaAccessor
from dataloaders.url_image_loader import URLImageLoader, create_captioned_image_download_loader
from utils import read_parquet_as_df
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
    parser.add_argument('--cache_dir', type=str, default='/data/jingdong')
    parser.add_argument('--datasource', type=str, default='datacomp')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--dataset_location', type=str, default='/data/jingdong/datacomp/small/metadata')
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
    
    image_loader = get_image_loader(args.datasource, args.dataset_location, args.num_samples, args.batch_size)
    print("Finished loading data loaders")

    all_generated_captions = []
    all_image_paths = []
    all_original_captions = []
    partition = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(image_loader)):
            images = batch["image"]
            if not images:
                continue
            image_paths = batch["image_path"]
            original_captions = batch['caption']

            captions_per_prompt = captioner.generate_caption(images)
                
            all_original_captions += original_captions
            all_image_paths += image_paths

            if not all_generated_captions:
                all_generated_captions = captions_per_prompt
            else:
                for i in range(len(all_generated_captions)):
                    all_generated_captions[i] += captions_per_prompt[i]

            if len(all_image_paths) > args.storage_size:
                filename = os.path.join(args.result_location, f"{file_prefix}_{partition}.tsv")
                save_captions(all_image_paths, all_original_captions, all_generated_captions, filename, prompts)
                
                partition += 1
        
        if all_original_captions:
            filename = os.path.join(args.result_location, f"{file_prefix}_{partition}.tsv")
            save_captions(all_image_paths, all_original_captions, all_generated_captions, filename, prompts)


def get_captioner(args):
    if args.captioner == 'blip2':
        return Blip2Accessor(device=args.device, cache_dir=args.cache_dir), None
    
    elif args.captioner == 'llava':
        return LlavaAccessor(args, device=args.device, cache_dir=args.cache_dir)
        
    else:
        raise NotImplementedError


def get_image_loader(datasource, dataset_location, num_samples, batch_size):
    if datasource == 'datacomp':
        df = read_parquet_as_df(dataset_location, num_sample=num_samples)
        urls = df["url"].tolist()
        original_captions = df["text"].tolist()

        return create_captioned_image_download_loader(urls, original_captions, batch_size)
    else:
        raise NotImplementedError


def save_captions(image_paths: list, original_captions: list, generated_captions: list, filename: str, prompts=None):
    columns = {
        'image_paths': image_paths,
        'original_captions': original_captions,
    }

    if prompts is None:
        caption_col_names = [f"generated_caption_group_{i}" for i in range(len(generated_captions))]
    else:
        assert len(prompts) == len(generated_captions)
        caption_col_names = prompts

    for i in range(len(caption_col_names)):
        columns[caption_col_names[i]] = generated_captions[i]

    pd.DataFrame(columns).to_csv(filename, index=False, sep='\t')
    
    image_paths.clear()
    original_captions.clear()
    generated_captions.clear()


if __name__ == "__main__":
    main()
