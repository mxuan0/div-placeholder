from utils import download_image
from torch.utils.data import Dataset


class CaptionedImageDownloadDataset(Dataset):
    def __init__(self, urls, captions):
        self.urls = urls
        self.captions = captions

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        
        return {
            "image_id": url,
            "image": download_image(url),
            "caption": self.captions[idx]
        }
