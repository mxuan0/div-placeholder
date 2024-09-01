from tqdm import tqdm
import pandas as pd
import os
import re
import glob
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def random_sample_from_parquet_in_dir(input_path, frac, output_path):
    dfs = []
    for file in tqdm(os.listdir(input_path)):
        if file.endswith('.parquet'):
            full_path = os.path.join(input_path, file)
            
            df = pd.read_parquet(full_path)
            dfs.append(df)

    filtered_df = pd.concat(dfs, ignore_index=True).sample(frac=frac)
    
    print(len(filtered_df))
    filtered_df.to_parquet(output_path)


def remove_extra_newlines_in_generated_captions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    processed_lines = [lines[0]]
    current_line = ""
    
    for line in lines[1:]:
        if re.match(r'^\d{12}', line):  # Check if the line starts with 12 digits
            if current_line:  # If there is content in current_line, add it to processed_lines
                processed_lines.append(current_line)
            current_line = line.strip()  # Start a new line
        else:
            current_line += ' ' + line.strip()  # Append to the previous line
    
    if current_line:  # Add the last line
        processed_lines.append(current_line)
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(processed_lines))


def normalize_whitespace(text):
    # Replace newline, tab, or multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()


def list_lambda(input_list: list, func):
    return [func(e) for e in input_list]


def filter_none_collate_fn(batch):
    images = []
    urls = []
    original_captions = []

    for item in batch:
        if item["image"] is not None:
            images.append(item["image"])
            urls.append(item["image_id"])
            original_captions.append(item["caption"])

    return {
        "image_id": urls,
        "image": images,
        "caption": original_captions
    }


def read_parquet_as_df(path: str, num_sample: int=None) -> pd.DataFrame:
    if os.path.isdir(path):        
        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
    elif os.path.isfile(path) and path.endswith('.parquet'):
        parquet_files = [path]

    if not parquet_files:
        print('path contains 0 parquet files')
        return None
    
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    
    if num_sample is None:
        return df
    return df.sample(n=num_sample)


def download_image(url, timeout=5):
    try:
        image = Image.open(requests.get(url, stream=True, timeout=timeout).raw)
        if image.mode != "RGB" and image.mode != "RGBA":
            return None 
        return image
    except Exception as e:
        # print(f"failed to download {url}")
        return None


def download_images_multithread(image_urls, max_workers=32, timeout=5):
    def _download_image(url):
        try:
            image = Image.open(requests.get(url, stream=True, timeout=timeout).raw)
            return image
        except Exception as e:
            return None

    images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = [executor.submit(_download_image, url) for url in image_urls]
        for future in as_completed(future_to_url):
            images.append(future.result())

    return [image for image in images if image is not None]
