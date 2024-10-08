from utils import download_image, filter_none_collate_fn
from torch.utils.data import DataLoader
from datasets.captioned_image_download_dataset import CaptionedImageDownloadDataset


class URLImageLoader:
    def __init__(self, urls, batch_size=32):
        self.urls = urls
        self.offset = 0
        self.batch_size = batch_size

    def next(self):
        batch = []
        valid_urls = []

        while self.offset < len(self.urls) and len(batch) < self.batch_size:
            url = self.urls[self.offset]
            image = download_image(url)

            if image is not None:
                batch.append(image)
                valid_urls.append(url)

            self.offset += 1

        return batch, valid_urls
    
    def has_next(self):
        return self.offset < len(self.urls)


def create_captioned_image_download_loader(urls, captions, batch_size, num_workers=2):
    dataset = CaptionedImageDownloadDataset(urls, captions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=filter_none_collate_fn, num_workers=num_workers)

    return dataloader
