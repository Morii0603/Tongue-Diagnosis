import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from build_model import ResNet50, ResNet101, Swin_T, efficientnet_v2_s
from engine import MyTrainer
from read_data import TongueDataset

epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_excel("Shexiang-2024-06.23.xlsx")


train_data, test_data = train_test_split(data, test_size=0.1,random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.125,random_state=42)
train_dataset = TongueDataset(train_data, transform=TongueDataset.get_tranforms(train=True))
val_dataset = TongueDataset(val_data, transform=TongueDataset.get_tranforms(train=False))
test_dataset = TongueDataset(test_data, transform=TongueDataset.get_tranforms(train=False))

train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False)
val_loader = DataLoader(dataset=val_dataset,batch_size=64,shuffle=False)
test_loader = DataLoader(dataset=val_dataset,batch_size=64,shuffle=False)

net = efficientnet_v2_s(out_features=1, pretrained=True)
net = net.to(device)


criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)

trainer = MyTrainer(net, optimizer, criterion, device)
trainer.fit(train_loader, val_loader, epochs)


