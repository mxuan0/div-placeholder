from plotting.plot_image_grid import plot_images_grid
from utils import download_images_multithread

import pandas as pd


path = '/data/jingdong/datacomp/small/generated_captions/pilot/blip2_partition_0.csv'
urls = pd.read_csv(path, sep='\t')['ids'].tolist()

images = download_images_multithread(urls)
plot_images_grid(images, cols=10, save_path='plotting/plots/pilot_images.png')  