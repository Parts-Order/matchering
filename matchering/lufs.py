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
from scipy import signal
from .log import debug


def _k_weighting_filter(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create K-weighting filter for LUFS measurement (ITU-R BS.1770-4).
    
    Args:
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    # Pre-filter (high-pass)
    f_h = 1681.974450955533
    G_h = 3.99984385397
    Q_h = 0.7071752369554196
    
    # RLB filter (high-shelf)
    f_b = 38.13547087602444
    G_b = 0.0
    Q_b = 0.5003270373238773
    
    # Convert to digital domain
    f_h_digital = 2 * np.pi * f_h / sample_rate
    f_b_digital = 2 * np.pi * f_b / sample_rate
    
    # Pre-filter coefficients
    K_h = np.tan(f_h_digital / 2)
    V_h = 10 ** (G_h / 20)
    
    b0_h = (1 + np.sqrt(V_h) * K_h / Q_h + V_h * K_h**2) / (1 + K_h / Q_h + K_h**2)
    b1_h = 2 * (V_h * K_h**2 - 1) / (1 + K_h / Q_h + K_h**2)
    b2_h = (1 - np.sqrt(V_h) * K_h / Q_h + V_h * K_h**2) / (1 + K_h / Q_h + K_h**2)
    a0_h = 1
    a1_h = 2 * (K_h**2 - 1) / (1 + K_h / Q_h + K_h**2)
    a2_h = (1 - K_h / Q_h + K_h**2) / (1 + K_h / Q_h + K_h**2)
    
    # RLB filter coefficients
    K_b = np.tan(f_b_digital / 2)
    V_b = 10 ** (G_b / 20)
    
    b0_b = (1 + np.sqrt(V_b) * K_b / Q_b + V_b * K_b**2) / (1 + K_b / Q_b + K_b**2)
    b1_b = 2 * (V_b * K_b**2 - 1) / (1 + K_b / Q_b + K_b**2)
    b2_b = (1 - np.sqrt(V_b) * K_b / Q_b + V_b * K_b**2) / (1 + K_b / Q_b + K_b**2)
    a0_b = 1
    a1_b = 2 * (K_b**2 - 1) / (1 + K_b / Q_b + K_b**2)
    a2_b = (1 - K_b / Q_b + K_b**2) / (1 + K_b / Q_b + K_b**2)
    
    # Cascade filters
    b_pre = np.array([b0_h, b1_h, b2_h])
    a_pre = np.array([a0_h, a1_h, a2_h])
    
    b_rlb = np.array([b0_b, b1_b, b2_b])
    a_rlb = np.array([a0_b, a1_b, a2_b])
    
    # Combine filters
    b_combined = np.convolve(b_pre, b_rlb)
    a_combined = np.convolve(a_pre, a_rlb)
    
    return b_combined, a_combined


def measure_lufs(audio: np.ndarray, sample_rate: int, gate_block_size: float = 0.4) -> float:
    """
    Measure integrated loudness in LUFS (ITU-R BS.1770-4).
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate in Hz
        gate_block_size: Block size for gating in seconds (default: 0.4s)
        
    Returns:
        Integrated loudness in LUFS
    """
    if len(audio.shape) == 1:
        audio = audio.reshape(-1, 1)
    
    debug(f"Measuring LUFS for {audio.shape[1]} channel audio at {sample_rate}Hz")
    
    # Apply K-weighting filter
    b, a = _k_weighting_filter(sample_rate)
    filtered = signal.lfilter(b, a, audio, axis=0)
    
    # Calculate block size in samples
    block_samples = int(gate_block_size * sample_rate)
    overlap_samples = int(0.75 * block_samples)  # 75% overlap
    hop_samples = block_samples - overlap_samples
    
    # Calculate loudness for each block
    num_blocks = (len(filtered) - block_samples) // hop_samples + 1
    block_loudness = []
    
    for i in range(num_blocks):
        start_idx = i * hop_samples
        end_idx = start_idx + block_samples
        
        if end_idx > len(filtered):
            break
            
        block = filtered[start_idx:end_idx]
        
        # Channel weighting (stereo: L=1.0, R=1.0, others depend on config)
        if block.shape[1] == 1:
            # Mono
            mean_square = np.mean(block[:, 0]**2)
        elif block.shape[1] == 2:
            # Stereo
            mean_square = np.mean(block[:, 0]**2) + np.mean(block[:, 1]**2)
        else:
            # Multi-channel (simplified - would need proper channel weighting)
            mean_square = np.sum(np.mean(block**2, axis=0))
        
        if mean_square > 0:
            loudness = -0.691 + 10 * np.log10(mean_square)
            block_loudness.append(loudness)
    
    if not block_loudness:
        debug("No valid blocks found for LUFS measurement")
        return -np.inf
    
    block_loudness = np.array(block_loudness)
    
    # First gating: -70 LUFS absolute threshold
    valid_blocks = block_loudness[block_loudness >= -70.0]
    
    if len(valid_blocks) == 0:
        debug("No blocks above -70 LUFS threshold")
        return -np.inf
    
    # Second gating: relative threshold (mean - 10 LUFS)
    relative_threshold = np.mean(valid_blocks) - 10.0
    gated_blocks = valid_blocks[valid_blocks >= relative_threshold]
    
    if len(gated_blocks) == 0:
        debug("No blocks above relative threshold")
        return -np.inf
    
    # Integrated loudness
    integrated_loudness = np.mean(gated_blocks)
    
    debug(f"LUFS measurement: {integrated_loudness:.1f} LUFS ({len(gated_blocks)} blocks)")
    return integrated_loudness


def measure_lufs_range(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure loudness range (LRA) in LU (ITU-R BS.1770-4).
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate in Hz
        
    Returns:
        Loudness range in LU
    """
    if len(audio.shape) == 1:
        audio = audio.reshape(-1, 1)
    
    # Apply K-weighting filter
    b, a = _k_weighting_filter(sample_rate)
    filtered = signal.lfilter(b, a, audio, axis=0)
    
    # Calculate short-term loudness (3s blocks with 10% overlap)
    block_samples = int(3.0 * sample_rate)
    hop_samples = int(0.1 * block_samples)
    
    num_blocks = (len(filtered) - block_samples) // hop_samples + 1
    short_term_loudness = []
    
    for i in range(num_blocks):
        start_idx = i * hop_samples
        end_idx = start_idx + block_samples
        
        if end_idx > len(filtered):
            break
            
        block = filtered[start_idx:end_idx]
        
        # Channel weighting
        if block.shape[1] == 1:
            mean_square = np.mean(block[:, 0]**2)
        elif block.shape[1] == 2:
            mean_square = np.mean(block[:, 0]**2) + np.mean(block[:, 1]**2)
        else:
            mean_square = np.sum(np.mean(block**2, axis=0))
        
        if mean_square > 0:
            loudness = -0.691 + 10 * np.log10(mean_square)
            short_term_loudness.append(loudness)
    
    if not short_term_loudness:
        return 0.0
    
    short_term_loudness = np.array(short_term_loudness)
    
    # Apply absolute and relative gating
    valid_blocks = short_term_loudness[short_term_loudness >= -70.0]
    
    if len(valid_blocks) == 0:
        return 0.0
    
    relative_threshold = np.mean(valid_blocks) - 20.0  # -20 LU for LRA
    gated_blocks = valid_blocks[valid_blocks >= relative_threshold]
    
    if len(gated_blocks) < 2:
        return 0.0
    
    # LRA is difference between 95th and 10th percentiles
    lra = np.percentile(gated_blocks, 95) - np.percentile(gated_blocks, 10)
    
    debug(f"Loudness range: {lra:.1f} LU")
    return lra


def normalize_to_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float = -18.0) -> tuple[np.ndarray, float]:
    """
    Normalize audio to target LUFS level.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate in Hz
        target_lufs: Target LUFS level (default: -18.0 for Atmos)
        
    Returns:
        Tuple of (normalized_audio, applied_gain_db)
    """
    current_lufs = measure_lufs(audio, sample_rate)
    
    if current_lufs == -np.inf:
        debug("Cannot normalize: no valid LUFS measurement")
        return audio, 0.0
    
    # Calculate required gain
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized_audio = audio * gain_linear
    
    debug(f"LUFS normalization: {current_lufs:.1f} -> {target_lufs:.1f} LUFS ({gain_db:+.1f} dB)")
    return normalized_audio, gain_db


def check_atmos_compliance(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Check Dolby Atmos compliance for audio.
    
    Args:
        audio: Input audio array (samples, channels)
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with compliance results
    """
    lufs = measure_lufs(audio, sample_rate)
    lra = measure_lufs_range(audio, sample_rate)
    
    # Dolby Atmos specifications
    lufs_compliant = -20.0 <= lufs <= -16.0  # -18 LUFS ±2
    lra_compliant = lra <= 20.0  # Maximum 20 LU
    
    # Calculate headroom to -18 LUFS
    headroom_to_18 = -18.0 - lufs if lufs != -np.inf else np.inf
    
    compliance = {
        "lufs": lufs,
        "lra": lra,
        "lufs_compliant": lufs_compliant,
        "lra_compliant": lra_compliant,
        "atmos_compliant": lufs_compliant and lra_compliant,
        "headroom_to_18_lufs": headroom_to_18,
        "recommended_gain_db": headroom_to_18 if abs(headroom_to_18) > 0.1 else 0.0
    }
    
    debug(f"Atmos compliance: LUFS={lufs:.1f} {'✓' if lufs_compliant else '✗'}, "
          f"LRA={lra:.1f} {'✓' if lra_compliant else '✗'}")
    
    return compliance