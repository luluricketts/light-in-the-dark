import glob
import os 

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LightDataset(Dataset):
   
    def __init__(self, cfg, cls, transform=None):

        self.images = glob.glob(os.path.join(cfg.data_dir, cls, cfg.ext))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image


def get_data_loader(cfg, cls):

    load_size = int(1.1 * cfg.image_size)
    train_transform = transforms.Compose([
        transforms.Resize([load_size, load_size], Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomCrop(cfg.image_size),
        transforms.RandomHorizontalFlip()
    ])
    dataset = LightDataset(
        cfg, cls, train_transform
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers
    )

    return dataloader
