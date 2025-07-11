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
from .oversampling import upsample, downsample, get_oversampling_factor, is_oversampling_needed


def detect_true_peaks(audio: np.ndarray, sample_rate: int, threshold: float = 0.0) -> np.ndarray:
    """
    Detect true peaks (intersample peaks) in audio signal.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        threshold: Peak threshold in linear scale (0.0 = 0dBFS)
        
    Returns:
        Boolean array indicating true peak locations
    """
    if not is_oversampling_needed(sample_rate):
        debug("No oversampling needed for true peak detection")
        return np.abs(audio).max(axis=1) > threshold
    
    # Oversample to detect intersample peaks
    factor = get_oversampling_factor(sample_rate)
    oversampled = upsample(audio, sample_rate, factor)
    
    # Detect peaks in oversampled signal
    peak_mask = np.abs(oversampled).max(axis=1) > threshold
    
    # Downsample peak mask to original resolution
    # Use max pooling to preserve peak information
    original_length = audio.shape[0]
    downsampled_mask = np.zeros(original_length, dtype=bool)
    
    for i in range(original_length):
        start_idx = i * factor
        end_idx = min((i + 1) * factor, len(peak_mask))
        downsampled_mask[i] = np.any(peak_mask[start_idx:end_idx])
    
    peak_count = np.sum(downsampled_mask)
    debug(f"Detected {peak_count} true peaks above threshold {threshold:.6f}")
    
    return downsampled_mask


def measure_true_peak_level(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure true peak level in audio signal.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        
    Returns:
        True peak level in dBFS
    """
    if not is_oversampling_needed(sample_rate):
        # Standard peak measurement
        peak_linear = np.abs(audio).max()
    else:
        # Oversample to measure intersample peaks
        factor = get_oversampling_factor(sample_rate)
        oversampled = upsample(audio, sample_rate, factor)
        peak_linear = np.abs(oversampled).max()
    
    # Convert to dBFS
    if peak_linear > 0:
        peak_db = 20 * np.log10(peak_linear)
    else:
        peak_db = -np.inf
    
    debug(f"True peak level: {peak_db:.2f} dBFS")
    return peak_db


def calculate_true_peak_gain_reduction(audio: np.ndarray, sample_rate: int, threshold_db: float) -> float:
    """
    Calculate gain reduction needed to prevent true peak clipping.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        threshold_db: True peak threshold in dBFS
        
    Returns:
        Gain reduction factor (linear, <= 1.0)
    """
    true_peak_db = measure_true_peak_level(audio, sample_rate)
    
    if true_peak_db <= threshold_db:
        debug("No true peak gain reduction needed")
        return 1.0
    
    # Calculate required gain reduction
    reduction_db = true_peak_db - threshold_db
    reduction_linear = 10 ** (-reduction_db / 20)
    
    debug(f"True peak gain reduction: {reduction_db:.2f} dB (factor: {reduction_linear:.6f})")
    return reduction_linear


def apply_true_peak_limiting(audio: np.ndarray, sample_rate: int, threshold_db: float = -0.1) -> np.ndarray:
    """
    Apply true peak limiting to prevent intersample clipping.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        threshold_db: True peak threshold in dBFS (default: -0.1dBFS)
        
    Returns:
        True peak limited audio
    """
    debug(f"Applying true peak limiting with threshold: {threshold_db:.1f} dBFS")
    
    # Calculate required gain reduction
    gain_reduction = calculate_true_peak_gain_reduction(audio, sample_rate, threshold_db)
    
    if gain_reduction >= 1.0:
        debug("No true peak limiting applied")
        return audio
    
    # Apply gain reduction
    limited_audio = audio * gain_reduction
    
    # Verify true peak level after limiting
    final_peak_db = measure_true_peak_level(limited_audio, sample_rate)
    debug(f"Final true peak level: {final_peak_db:.2f} dBFS")
    
    return limited_audio


def get_true_peak_headroom(audio: np.ndarray, sample_rate: int) -> float:
    """
    Calculate available headroom before true peak clipping.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        
    Returns:
        Available headroom in dB
    """
    true_peak_db = measure_true_peak_level(audio, sample_rate)
    headroom_db = 0.0 - true_peak_db
    
    debug(f"True peak headroom: {headroom_db:.2f} dB")
    return headroom_db


def is_true_peak_compliant(audio: np.ndarray, sample_rate: int, standard: str = "EBU") -> bool:
    """
    Check if audio is compliant with broadcast standards for true peak levels.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate
        standard: Broadcast standard ("EBU", "ATSC", "ARIB")
        
    Returns:
        True if compliant with standard
    """
    true_peak_db = measure_true_peak_level(audio, sample_rate)
    
    # Standard true peak limits
    limits = {
        "EBU": -1.0,    # EBU R128 (European Broadcasting Union)
        "ATSC": -3.0,   # ATSC A/85 (North American broadcast)
        "ARIB": -1.0    # ARIB TR-B32 (Japanese broadcast)
    }
    
    if standard not in limits:
        debug(f"Unknown standard: {standard}, using EBU limit")
        standard = "EBU"
    
    limit_db = limits[standard]
    compliant = true_peak_db <= limit_db
    
    debug(f"True peak compliance ({standard}): {compliant} "
          f"(level: {true_peak_db:.2f} dBFS, limit: {limit_db:.1f} dBFS)")
    
    return compliant