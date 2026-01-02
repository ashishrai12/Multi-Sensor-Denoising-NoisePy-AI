# NoisePy Sensor Analysis

Characterizing noise in multi-sensor systems using principles of NoisePy: Spectral Whitening, One-Bit Normalization, and Cross-Correlation.

## Overview
This project simulates a dual-sensor environment to study how ambient correlated noise (representing a traveling signal) can be extracted from a background of electronic system noise and non-Gaussian glitches.

### Noise Components Simulated:
1.  **Correlated 'Ambient' Noise**: Common signal source with a sample delay between sensors.
2.  **Uncorrelated 'System' Noise**: Independent Gaussian noise per sensor.
3.  **Non-Gaussian Spikes**: Random high-amplitude glitches.

## Processing Workflow
The `noise_analysis.py` script implements the following NoisePy-inspired steps:
- **Spectral Whitening**: Normalizes the amplitude spectrum to ensure all frequencies contribute equally to the correlation, effectively flattening the noise floor.
- **One-Bit Normalization**: Replaces the signal with its sign (-1 or 1). This is extremely robust against high-amplitude spikes (glitches) which would otherwise dominate the cross-correlation.
- **Cross-Correlation**: Computes the coherent response between sensors to find the time-of-flight (delay).

## Visual Results
### 1. Multi-Panel Dashboard (`noise_dashboard.png`)
- **Raw vs. Processed**: Shows how One-Bit normalization "squashes" the signal and removes the influence of spikes.
- **PSD Analysis**: Demonstrates how Whitening results in a flat Power Spectral Density.
- **System Characterization**: Displays the recovered cross-correlation pulse, accurately identifying the true delay despite the noise.

### 2. SNR Degradation (`degradation_heatmap.png`)
A waterfall plot showing the cross-correlation peak as the "System Noise" level increases. It illustrates the robustness of the NoisePy workflow and the point at which the coherent signal is lost in the noise.

## Installation & Usage
```bash
pip install numpy scipy matplotlib
python noise_analysis.py
```

## Results Summary
The implementation successfully recovers the simulated 100-sample delay. Even as the system noise standard deviation increases beyond the ambient signal level, the whitening and one-bit normalization preserve the phase information necessary for a strong correlation peak.
