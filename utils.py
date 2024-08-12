import pandas as pd
import os
import glob
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    return images
