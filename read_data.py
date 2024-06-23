import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class TongueDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        id = int(row["TXM"])
        img = Image.open(f"final/{id}.JPG").convert("RGB")
        label = row["OP"]
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor([label])
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def get_tranforms(train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # transforms.RandomErasing()
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
