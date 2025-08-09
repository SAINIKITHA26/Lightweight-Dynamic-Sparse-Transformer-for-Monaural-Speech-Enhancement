import sys
import os
import time
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QWidget, QProgressBar,
                            QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import soundfile as sf
import tempfile
from pydub import AudioSegment
import librosa
from math import log10

# --- Audio Enhancer class ---
class AudioEnhancer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = torch.hub.load(
                'facebookresearch/denoiser',
                'dns64',
                pretrained=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(None, "Model Error", f"Failed to load denoiser: {str(e)}")
            raise

    def enhance_audio(self, noisy_samples, sr):
        try:
            start_time = time.time()
            
            if sr != 16000:
                noisy_samples = librosa.resample(noisy_samples, orig_sr=sr, target_sr=16000)
                sr = 16000

            noisy_tensor = torch.from_numpy(noisy_samples).float().to(self.device)
            
            if len(noisy_tensor.shape) == 1:
                noisy_tensor = noisy_tensor.unsqueeze(0)
            
            with torch.no_grad():
                enhanced_tensor = self.model(noisy_tensor)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return enhanced_tensor.squeeze().cpu().numpy(), sr, processing_time
        except Exception as e:
            QMessageBox.critical(None, "Processing Error", f"Audio enhancement failed: {str(e)}")
            raise

# --- Worker Thread ---
class Worker(QThread):
    finished = pyqtSignal(np.ndarray, int, np.ndarray, int, float)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, enhancer, file_path):
        super().__init__()
        self.enhancer = enhancer
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress.emit(10)
            audio = AudioSegment.from_file(self.file_path)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            sr = audio.frame_rate
            
            self.progress.emit(40)
            
            enhanced_samples, sr, processing_time = self.enhancer.enhance_audio(samples, sr)
            
            self.progress.emit(80)
            self.finished.emit(samples, sr, enhanced_samples, sr, processing_time)
            self.progress.emit(100)
            
        except Exception as e:
            self.error.emit(str(e))
            self.progress.emit(0)

# --- Spectrogram Display Window ---
class SpectrogramWindow(QWidget):
    def __init__(self, noisy_samples, enhanced_samples, sr):
        super().__init__()
        self.setWindowTitle("Spectrograms")
        self.setGeometry(150, 150, 1000, 500)
        
        layout = QVBoxLayout()
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax1.specgram(noisy_samples, Fs=sr, NFFT=1024, noverlap=512, cmap='inferno')
        self.ax1.set_title("Noisy Audio Spectrogram")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Frequency (Hz)")
        
        self.ax2.specgram(enhanced_samples, Fs=sr, NFFT=1024, noverlap=512, cmap='inferno')
        self.ax2.set_title("Enhanced Audio Spectrogram")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Frequency (Hz)")
        
        self.figure.tight_layout()
        self.canvas.draw()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lightweight Dynamic Sparse Transformer for Monaural Speech Enhancement")
        self.setGeometry(100, 100, 800, 600)
        
        try:
            self.enhancer = AudioEnhancer()
        except:
            sys.exit(1)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_button = QPushButton("Select Audio File")
        file_button.clicked.connect(self.open_file_dialog)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_button)
        layout.addLayout(file_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Action buttons
        button_layout = QHBoxLayout()
        self.play_noisy_button = QPushButton("Play Noisy Audio")
        self.play_noisy_button.setEnabled(False)
        self.play_noisy_button.clicked.connect(self.play_noisy)
        button_layout.addWidget(self.play_noisy_button)
        
        self.play_enhanced_button = QPushButton("Play Enhanced Audio")
        self.play_enhanced_button.setEnabled(False)
        self.play_enhanced_button.clicked.connect(self.play_enhanced)
        button_layout.addWidget(self.play_enhanced_button)
        
        self.save_button = QPushButton("Save Enhanced Audio")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_enhanced)
        button_layout.addWidget(self.save_button)

        self.spectrogram_button = QPushButton("Show Spectrograms")
        self.spectrogram_button.setEnabled(False)
        self.spectrogram_button.clicked.connect(self.show_spectrogram)
        button_layout.addWidget(self.spectrogram_button)

        layout.addLayout(button_layout)
        
        # Stats label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)
        
        # Waveform display
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Audio player
        self.audio_player = None
        self.noisy_samples = None
        self.enhanced_samples = None
        self.sr = None
        
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.process_audio(file_path)
    
    def process_audio(self, file_path):
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.play_noisy_button.setEnabled(False)
        self.play_enhanced_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.spectrogram_button.setEnabled(False)
        
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()
        
        self.worker = Worker(self.enhancer, file_path)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def on_processing_finished(self, noisy_samples, sr, enhanced_samples, _, processing_time):
        self.noisy_samples = noisy_samples
        self.enhanced_samples = enhanced_samples
        self.sr = sr
        self.processing_time = processing_time
        
        self.play_noisy_button.setEnabled(True)
        self.play_enhanced_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.spectrogram_button.setEnabled(True)
        self.progress.setVisible(False)
        
        time_axis = np.arange(len(noisy_samples)) / sr
        
        self.ax1.clear()
        self.ax1.plot(time_axis, noisy_samples)
        self.ax1.set_title("Noisy Audio (Waveform)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        
        self.ax2.clear()
        self.ax2.plot(time_axis, enhanced_samples, color='orange')
        self.ax2.set_title("Enhanced Audio (Waveform)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Amplitude")
        
        self.figure.tight_layout()
        self.canvas.draw()
        def compute_snr(clean, enhanced):
            noise = clean - enhanced
            snr = 10 * log10(np.sum(clean ** 2) / np.sum(noise ** 2))
            return snr
        
        self.show_stats()
        try:
            snr_value = compute_snr(self.noisy_samples, self.enhanced_samples)
            self.stats_label.setText(
                self.stats_label.text() +
                f"<br><b>Estimated SNR Improvement:</b> {snr_value:.2f} dB"
            )
        except Exception as e:
            print("SNR calculation error:", e)
            
    def show_stats(self):
        noisy_power = np.mean(self.noisy_samples**2)
        enhanced_power = np.mean(self.enhanced_samples**2)
        noise_reduction = (noisy_power - enhanced_power) / noisy_power * 100
        
        self.stats_label.setText(
            f"<b>Processing Time:</b> {self.processing_time:.2f} sec<br>"
            f"<b>Original Noise Power:</b> {noisy_power:.6f}<br>"
            f"<b>Enhanced Noise Power:</b> {enhanced_power:.6f}<br>"
            f"<b>Noise Reduction:</b> {noise_reduction:.2f}%"
        )
    
    def show_error(self, message):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", message)
    
    def play_noisy(self):
        self.play_audio(self.noisy_samples, self.sr)
    
    def play_enhanced(self):
        self.play_audio(self.enhanced_samples, self.sr)
    
    def play_audio(self, samples, sr):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                sf.write(tmpfile.name, samples, sr)
                if self.audio_player:
                    self.audio_player.stop()
                self.audio_player = AudioSegment.from_file(tmpfile.name)
                self.audio_player.export(tmpfile.name, format="wav")
                os.system(f"start {tmpfile.name}" if os.name == 'nt' else f"aplay {tmpfile.name}")
        except Exception as e:
            self.show_error(f"Playback error: {str(e)}")
    
    def save_enhanced(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Audio", "", 
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if file_path:
            if not file_path.lower().endswith('.wav'):
                file_path += '.wav'
            try:
                sf.write(file_path, self.enhanced_samples, self.sr)
                QMessageBox.information(self, "Success", "Audio saved successfully")
            except Exception as e:
                self.show_error(f"Save failed: {str(e)}")

    def show_spectrogram(self):
        self.spectrogram_window = SpectrogramWindow(self.noisy_samples, self.enhanced_samples, self.sr)
        self.spectrogram_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        import torchaudio
    except ImportError:
        QMessageBox.critical(None, "Error", "Please install torchaudio: pip install torchaudio")
        sys.exit(1)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
