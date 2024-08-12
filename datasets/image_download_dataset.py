from utils import download_image
from torch.utils.data import Dataset


class ImageDownloadDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        
        return {
            "id": url,
            "image": download_image(url)
        }
