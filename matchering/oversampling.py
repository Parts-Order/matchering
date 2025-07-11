# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from .log import debug

try:
    import soxr
    SOXR_AVAILABLE = True
except ImportError:
    SOXR_AVAILABLE = False
    from scipy import signal


def upsample(audio: np.ndarray, sample_rate: int, factor: int = 4) -> np.ndarray:
    """
    Upsample audio using high-quality SoX resampler.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Original sample rate
        factor: Upsampling factor (default: 4x)
        
    Returns:
        Upsampled audio array
    """
    if not SOXR_AVAILABLE:
        debug("SoX resampler not available, falling back to scipy")
        return _upsample_scipy(audio, sample_rate, factor)
    
    target_sr = sample_rate * factor
    debug(f"Upsampling {sample_rate}Hz -> {target_sr}Hz (factor: {factor}x)")
    
    # SoX resampler with Very High Quality settings
    upsampled = soxr.resample(
        audio, 
        sample_rate, 
        target_sr, 
        quality='VHQ'  # Very High Quality
    )
    
    debug(f"Upsampled from {audio.shape[0]} to {upsampled.shape[0]} samples")
    return upsampled


def downsample(audio: np.ndarray, sample_rate: int, factor: int = 4) -> np.ndarray:
    """
    Downsample audio using high-quality SoX resampler.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Current sample rate (should be upsampled rate)
        factor: Downsampling factor (default: 4x)
        
    Returns:
        Downsampled audio array
    """
    if not SOXR_AVAILABLE:
        debug("SoX resampler not available, falling back to scipy")
        return _downsample_scipy(audio, sample_rate, factor)
    
    target_sr = sample_rate // factor
    debug(f"Downsampling {sample_rate}Hz -> {target_sr}Hz (factor: {factor}x)")
    
    # SoX resampler with Very High Quality settings
    downsampled = soxr.resample(
        audio, 
        sample_rate, 
        target_sr, 
        quality='VHQ'  # Very High Quality
    )
    
    debug(f"Downsampled from {audio.shape[0]} to {downsampled.shape[0]} samples")
    return downsampled


def _upsample_scipy(audio: np.ndarray, sample_rate: int, factor: int) -> np.ndarray:
    """
    Fallback upsampling using scipy (lower quality).
    """
    # Zero-pad between samples
    upsampled = np.zeros((audio.shape[0] * factor, audio.shape[1]))
    upsampled[::factor] = audio
    
    # Anti-aliasing filter
    nyquist = sample_rate / 2
    sos = signal.butter(8, nyquist, fs=sample_rate * factor, output='sos')
    return signal.sosfilt(sos, upsampled, axis=0)


def _downsample_scipy(audio: np.ndarray, sample_rate: int, factor: int) -> np.ndarray:
    """
    Fallback downsampling using scipy (lower quality).
    """
    # Anti-aliasing filter before decimation
    nyquist = sample_rate / (2 * factor)
    sos = signal.butter(8, nyquist, fs=sample_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, audio, axis=0)
    
    # Decimate
    return filtered[::factor]


def get_oversampling_factor(sample_rate: int) -> int:
    """
    Determine appropriate oversampling factor based on sample rate.
    
    Args:
        sample_rate: Original sample rate
        
    Returns:
        Recommended oversampling factor
    """
    if sample_rate <= 48000:
        return 4  # 4x oversampling for standard rates
    elif sample_rate <= 96000:
        return 2  # 2x oversampling for high rates
    else:
        return 1  # No oversampling for very high rates


def is_oversampling_needed(sample_rate: int, enable_true_peak: bool = True) -> bool:
    """
    Determine if oversampling is needed based on configuration.
    
    Args:
        sample_rate: Current sample rate
        enable_true_peak: Whether true peak limiting is enabled
        
    Returns:
        True if oversampling should be applied
    """
    if not enable_true_peak:
        return False
    
    # Always oversample for true peak limiting at standard rates
    return sample_rate <= 96000