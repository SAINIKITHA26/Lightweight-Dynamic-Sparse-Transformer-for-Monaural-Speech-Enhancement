import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import soundfile as sf
import io
from PIL import Image

def plot_waveform(samples, sr, title="Waveform"):
    """Plot waveform from audio samples"""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(samples, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt.gcf()

def plot_spectrogram(magnitude, sr, hop_length, title="Spectrogram"):
    """Plot spectrogram from magnitude spectrogram"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(magnitude)),
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_comparison(noisy_samples, enhanced_samples, sr, hop_length=128):
    """Plot comparison between noisy and enhanced audio"""
    plt.figure(figsize=(12, 8))
    
    # Waveform comparison
    plt.subplot(2, 2, 1)
    librosa.display.waveshow(noisy_samples, sr=sr, alpha=0.5)
    plt.title("Noisy Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 2, 2)
    librosa.display.waveshow(enhanced_samples, sr=sr, color='orange')
    plt.title("Enhanced Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Spectrogram comparison
    plt.subplot(2, 2, 3)
    noisy_mag = librosa.stft(noisy_samples, hop_length=hop_length)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(noisy_mag)),
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Noisy Spectrogram")
    
    plt.subplot(2, 2, 4)
    enhanced_mag = librosa.stft(enhanced_samples, hop_length=hop_length)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(enhanced_mag)),
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Enhanced Spectrogram")
    
    plt.tight_layout()
    return plt.gcf()

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img