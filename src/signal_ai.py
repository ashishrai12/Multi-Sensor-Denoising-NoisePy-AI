import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SignalAutoencoder(nn.Module):
    def __init__(self, input_size=100):
        super(SignalAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_denoiser(signals, input_size=100, epochs=20, lr=0.001):
    """
    Trains a simple autoencoder on snippets of the signal.
    signals: list of numpy arrays (the 'clean' or 'normal' signals)
    """
    # Prepare data
    data = []
    for s in signals:
        # Break signal into windows
        for i in range(0, len(s) - input_size, input_size // 2):
            window = s[i:i+input_size]
            if len(window) == input_size:
                data.append(window)
    
    data = np.array(data)
    # Normalize
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-8)
    
    tensor_data = torch.FloatTensor(data)
    
    model = SignalAutoencoder(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model, mean, std

def detect_anomalies(model, signal, mean, std, input_size=100, threshold_multiplier=3.0):
    """
    Uses the trained autoencoder to detect anomalies (spikes).
    """
    model.eval()
    signal_norm = (signal - mean) / (std + 1e-8)
    
    windows = []
    offsets = []
    for i in range(0, len(signal_norm) - input_size, 10): # Overlap
        windows.append(signal_norm[i:i+input_size])
        offsets.append(i)
        
    windows = torch.FloatTensor(np.array(windows))
    with torch.no_grad():
        reconstructed = model(windows).numpy()
    
    # Calculate reconstruction error per window
    errors = np.mean((windows.numpy() - reconstructed)**2, axis=1)
    
    # Map errors back to the original signal length
    anomaly_score = np.zeros(len(signal))
    counts = np.zeros(len(signal))
    
    for i, offset in enumerate(offsets):
        anomaly_score[offset:offset+input_size] += errors[i]
        counts[offset:offset+input_size] += 1
        
    anomaly_score[counts > 0] /= counts[counts > 0]
    
    threshold = np.mean(anomaly_score) + threshold_multiplier * np.std(anomaly_score)
    is_anomaly = anomaly_score > threshold
    
    return is_anomaly, anomaly_score

def ai_denoise(model, signal, mean, std, input_size=100):
    """
    Attempts to denoise the signal using the autoencoder.
    """
    model.eval()
    signal_norm = (signal - mean) / (std + 1e-8)
    
    denoised = np.zeros(len(signal))
    counts = np.zeros(len(signal))
    
    for i in range(0, len(signal_norm) - input_size, input_size // 4):
        window = signal_norm[i:i+input_size]
        if len(window) < input_size: break
        
        window_tensor = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            rec = model(window_tensor).squeeze(0).numpy()
            
        denoised[i:i+input_size] += rec
        counts[i:i+input_size] += 1
        
    denoised[counts > 0] /= counts[counts > 0]
    # De-normalize
    denoised = denoised * std + mean
    return denoised
