import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from models.model_architecture import LightweightSparseTransformer
from utils.audio_processing import stft

config = {
    'sample_rate': 16000,
    'n_fft': 256,
    'hop_length': 64,
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-4,
    'model_dir': 'models',
    'pretrained_path': 'models/pretrained_weights.pth'
}

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, duration=1.0):
        self.num_samples = num_samples
        self.duration = duration
        self.sr = config['sample_rate']
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.normal(size=clean.shape)
        noisy = clean + noise
        
        clean_mag, _ = stft(clean)
        noisy_mag, _ = stft(noisy)
        return torch.FloatTensor(noisy_mag), torch.FloatTensor(clean_mag)

def train():
    os.makedirs(config['model_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightSparseTransformer().to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    dataset = SyntheticDataset(num_samples=1000)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        progress = tqdm(loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for noisy, clean in progress:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['pretrained_path'])
            print(f'Saved best model with loss: {best_loss:.4f}')

if __name__ == '__main__':
    train()