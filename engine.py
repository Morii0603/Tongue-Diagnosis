import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
class MyTrainer:
    def __init__(self, model, optimizer, criterion, device, save_path = "checkpoints") -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.total_train_loss = []
        self.total_valid_loss = []
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)
    def train_epoch(self, train_loader: DataLoader, epoch):
        self.model.train()
        train_loss = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for images, targets in train_pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            train_pbar.set_postfix(loss=loss.item())
        epoch_train_loss = np.mean(train_loss)
        self.total_train_loss.append(epoch_train_loss)
        return epoch_train_loss

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = []
        val_outputs, val_labels = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                val_loss.append(loss.item())
                val_outputs.extend(outputs.detach().cpu().numpy().tolist())
                val_labels.extend(targets.detach().cpu().numpy().tolist())
        epoch_val_loss = np.mean(val_loss)
        auc = roc_auc_score(val_labels, val_outputs)
        self.total_valid_loss.append(epoch_val_loss)
        return epoch_val_loss, auc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs):
        self.epochs = epochs
        min_val_loss = np.inf
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch)
            epoch_val_loss, auc = self.validate(val_loader)
            print(f"Epoch {epoch + 1}: Val Loss: {epoch_val_loss}, AUC: {auc}")
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), f"{self.save_path}/best.pth")
