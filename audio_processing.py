import numpy as np
import scipy.signal as signal
import soundfile as sf
from pydub import AudioSegment
import torch

def load_audio(file_path, target_sr=16000):
    """Load audio file and resample to target sample rate"""
    audio = AudioSegment.from_file(file_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples, target_sr

def save_audio(file_path, samples, sr):
    """Save audio to file"""
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32768.0).astype(np.int16)
    sf.write(file_path, samples, sr)

def stft(samples, n_fft=512, hop_length=128):
    """Compute STFT"""
    f, t, Zxx = signal.stft(samples, fs=16000, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    return mag, phase, t

def istft(mag, phase, hop_length=128):
    """Compute inverse STFT"""
    Zxx = mag * np.exp(1j * phase)
    _, samples = signal.istft(Zxx, fs=16000, noverlap=512-hop_length)
    return samples

def preprocess_audio(samples, device='cpu'):
    """Convert audio samples to network input"""
    mag, phase, _ = stft(samples)
    mag_tensor = torch.from_numpy(mag).unsqueeze(0).to(device)
    phase_tensor = torch.from_numpy(phase).unsqueeze(0).to(device)
    return mag_tensor, phase_tensor

def postprocess_audio(mag, phase, device='cpu'):
    """Convert network output to audio samples"""
    mag = mag.squeeze(0).cpu().numpy()
    phase = phase.squeeze(0).cpu().numpy()
    return istft(mag, phase)