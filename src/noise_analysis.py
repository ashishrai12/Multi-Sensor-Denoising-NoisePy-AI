import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, welch, hilbert
import os
try:
    import signal_ai
    HAS_AI = True
except ImportError:
    HAS_AI = False

def generate_noise_data(duration=10.0, fs=1000, ambient_std=1.0, system_std=0.5, spike_prob=0.005, spike_amp=10.0, delay_samples=50):
    """
    Simulates noise from two sensors.
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # 1. Correlated 'Ambient' Noise (Band-limited white noise)
    # We'll create a common source and shift it for the second sensor
    ambient_common = np.random.normal(0, ambient_std, n_samples + delay_samples)
    
    # Sensor 1 sees the ambient noise
    s1_ambient = ambient_common[:n_samples]
    # Sensor 2 sees it with a delay
    s2_ambient = ambient_common[delay_samples:]
    
    # 2. Uncorrelated 'System' Noise
    s1_system = np.random.normal(0, system_std, n_samples)
    s2_system = np.random.normal(0, system_std, n_samples)
    
    # 3. Non-Gaussian Spikes
    s1_spikes = np.zeros(n_samples)
    s2_spikes = np.zeros(n_samples)
    
    spikes1 = np.random.random(n_samples) < spike_prob
    spikes2 = np.random.random(n_samples) < spike_prob
    
    s1_spikes[spikes1] = np.random.choice([-1, 1], size=np.sum(spikes1)) * spike_amp
    s2_spikes[spikes2] = np.random.choice([-1, 1], size=np.sum(spikes2)) * spike_amp
    
    # Combine signals
    s1 = s1_ambient + s1_system + s1_spikes
    s2 = s2_ambient + s2_system + s2_spikes
    
    return t, s1, s2

def spectral_whitening(data, fs, low=10, high=250):
    """
    Applies spectral whitening to flatten the frequency content.
    """
    n = len(data)
    spec = np.fft.rfft(data, n=n)
    freq = np.fft.rfftfreq(n, d=1/fs)
    
    # Smoothing the amplitude spectrum
    amp = np.abs(spec)
    # Avoid division by zero
    amp[amp == 0] = 1e-10
    
    # Whiten
    whitened_spec = spec / amp
    
    # Bandpass filter within the whitening process
    mask = (freq >= low) & (freq <= high)
    whitened_spec[~mask] = 0
    
    return np.fft.irfft(whitened_spec, n=n)

def one_bit_normalization(data):
    """
    Apply one-bit normalization to mitigate non-Gaussian spikes.
    """
    return np.sign(data)

def cross_correlate(s1, s2):
    """
    Perform cross-correlation to extract coherent signals.
    """
    # Use 'same' to keep output size manageable and centered
    corr = correlate(s1, s2, mode='same')
    # Normalize by number of samples
    corr /= len(s1)
    return corr

def run_analysis():
    fs = 1000
    duration = 5.0
    delay = 100 # samples
    
    # Generate data
    t, s1_raw, s2_raw = generate_noise_data(duration=duration, fs=fs, delay_samples=delay)
    
    # Process Data
    # 1. Whitening
    s1_white = spectral_whitening(s1_raw, fs)
    s2_white = spectral_whitening(s2_raw, fs)
    
    # 2. One-bit
    s1_proc = one_bit_normalization(s1_white)
    s2_proc = one_bit_normalization(s2_white)
    
    # 3. Correlation
    corr_raw = cross_correlate(s1_raw, s2_raw)
    corr_proc = cross_correlate(s1_proc, s2_proc)
    
    lags = np.arange(-len(corr_raw)//2, len(corr_raw)//2)
    
    # --- AI COMPONENT ---
    s1_ai_denoised = None
    anomalies_s1 = None
    if HAS_AI:
        print("Training AI Signal Model...")
        # Generate clean data for training (no spikes)
        _, train_s1, train_s2 = generate_noise_data(duration=10.0, fs=fs, spike_prob=0)
        model, mean, std = signal_ai.train_denoiser([train_s1, train_s2], epochs=15)
        
        print("Applying AI Analysis...")
        s1_ai_denoised = signal_ai.ai_denoise(model, s1_raw, mean, std)
        anomalies_s1, _ = signal_ai.detect_anomalies(model, s1_raw, mean, std)
        
        # Cross-correlate with AI-denoised signal
        s2_ai_denoised = signal_ai.ai_denoise(model, s2_raw, mean, std)
        corr_ai = cross_correlate(s1_ai_denoised, s2_ai_denoised)
    # ---------------------
    
    # Visual Characterization Dashboard
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)
    
    # Panel 1: Raw vs Processed Time Series
    axes[0, 0].plot(t[:500], s1_raw[:500], color='#ff4b5c', alpha=0.8, label='Raw Signal')
    axes[0, 0].set_title('Raw Time Series (Sensor 1 - Snippet)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t[:500], s1_proc[:500], color='#1f77b4', alpha=0.8, label='1-Bit + Whitened')
    axes[0, 1].set_title('Processed Time Series (Sensor 1 - Snippet)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 2: PSD Analysis
    f_raw, p_raw = welch(s1_raw, fs, nperseg=256)
    f_white, p_white = welch(s1_white, fs, nperseg=256)
    
    axes[1, 0].semilogy(f_raw, p_raw, color='#ff4b5c', label='Raw PSD')
    axes[1, 0].set_title('Power Spectral Density (Raw)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('V^2/Hz')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(f_white, p_white, color='#2ca02c', label='Whitened PSD')
    axes[1, 1].set_title('Power Spectral Density (Whitened)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Panel 3: Cross-Correlation results
    axes[2, 0].plot(lags, corr_raw, color='#ff7f0e', label='Raw Corr')
    axes[2, 0].set_title('Cross-Correlation (Raw)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Lag (samples)')
    axes[2, 0].grid(True, alpha=0.3)
    
    center = len(lags) // 2
    window = 500
    axes[2, 1].plot(lags[center-window:center+window], corr_proc[center-window:center+window], color='#9467bd', label='NoisePy Processed')
    
    if HAS_AI:
        axes[2, 1].plot(lags[center-window:center+window], corr_ai[center-window:center+window], color='#00d1b2', label='AI Denoised', alpha=0.7)
        
    axes[2, 1].axvline(delay, color='red', linestyle='--', alpha=0.7, label=f'True Delay ({delay})')
    axes[2, 1].set_title('System Response (Processed vs AI)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Lag (samples)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # New AI-specific visualization if AI is enabled
    if HAS_AI:
        fig_ai, ax_ai = plt.subplots(2, 1, figsize=(15, 8))
        ax_ai[0].plot(t[:1000], s1_raw[:1000], color='gray', alpha=0.5, label='Raw Signal')
        ax_ai[0].plot(t[:1000], s1_ai_denoised[:1000], color='#00d1b2', label='AI Denoised Output')
        ax_ai[0].set_title('AI Denoising Performance', fontsize=12, fontweight='bold')
        ax_ai[0].legend()
        
        # Show where anomalies were detected
        anomaly_times = t[:1000][anomalies_s1[:1000]]
        anomaly_vals = s1_raw[:1000][anomalies_s1[:1000]]
        ax_ai[1].plot(t[:1000], s1_raw[:1000], color='gray', alpha=0.5)
        ax_ai[1].scatter(anomaly_times, anomaly_vals, color='red', s=20, label='AI Detected Spikes')
        ax_ai[1].set_title('AI Anomaly Detection (Spikes)', fontsize=12, fontweight='bold')
        ax_ai[1].legend()
        plt.tight_layout()
        plt.savefig('ai_analysis_dashboard.png', dpi=150)
        plt.close(fig_ai)

    plt.tight_layout()
    plt.savefig('noise_dashboard.png', dpi=150)
    plt.close(fig)

    # SNR Degradation Loop
    print("Running SNR degradation analysis...")
    system_noise_levels = np.linspace(0.1, 5.0, 20)
    waterfall_data = []
    
    for level in system_noise_levels:
        _, sn1, sn2 = generate_noise_data(duration=2.0, fs=fs, system_std=level, delay_samples=delay)
        p1 = one_bit_normalization(spectral_whitening(sn1, fs))
        p2 = one_bit_normalization(spectral_whitening(sn2, fs))
        c = cross_correlate(p1, p2)
        
        # Focus on the peak area
        center = len(c) // 2
        waterfall_data.append(c[center-200:center+200])
        
    waterfall_data = np.array(waterfall_data)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(waterfall_data, aspect='auto', extent=[-200, 200, system_noise_levels[-1], system_noise_levels[0]], cmap='magma')
    plt.colorbar(label='Correlation Amplitude')
    plt.title('Correlation Peak Degradation vs System Noise', fontsize=14, fontweight='bold')
    plt.xlabel('Lag (samples)')
    plt.ylabel('System Noise Level (Std Dev)')
    plt.savefig('degradation_heatmap.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    run_analysis()
