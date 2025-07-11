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
from scipy.stats import kurtosis, skew
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from .log import debug
from .lufs import measure_lufs, measure_lufs_range, check_atmos_compliance
from .true_peak import measure_true_peak_level


def measure_rms(audio: np.ndarray, window_size: float = None, sample_rate: int = None) -> float:
    """
    Measure RMS level of audio.
    
    Args:
        audio: Input audio array
        window_size: Window size in seconds (if None, uses entire signal)
        sample_rate: Sample rate (required if window_size specified)
        
    Returns:
        RMS level in dB
    """
    if window_size is not None and sample_rate is not None:
        # Windowed RMS
        window_samples = int(window_size * sample_rate)
        rms_values = []
        
        for i in range(0, len(audio) - window_samples, window_samples // 2):
            window = audio[i:i + window_samples]
            rms = np.sqrt(np.mean(window**2))
            if rms > 0:
                rms_values.append(20 * np.log10(rms))
        
        return np.mean(rms_values) if rms_values else -np.inf
    else:
        # Overall RMS
        rms = np.sqrt(np.mean(audio**2))
        return 20 * np.log10(rms) if rms > 0 else -np.inf


def measure_peak_level(audio: np.ndarray) -> float:
    """
    Measure peak level of audio.
    
    Args:
        audio: Input audio array
        
    Returns:
        Peak level in dB
    """
    peak = np.abs(audio).max()
    return 20 * np.log10(peak) if peak > 0 else -np.inf


def measure_crest_factor(audio: np.ndarray) -> float:
    """
    Measure crest factor (peak-to-RMS ratio).
    
    Args:
        audio: Input audio array
        
    Returns:
        Crest factor in dB
    """
    peak_db = measure_peak_level(audio)
    rms_db = measure_rms(audio)
    
    if peak_db == -np.inf or rms_db == -np.inf:
        return np.inf
    
    return peak_db - rms_db


def measure_dynamic_range(audio: np.ndarray, sample_rate: int, method: str = "dr14") -> float:
    """
    Measure dynamic range using various methods.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        method: Method to use ("dr14", "lra", "crest_factor")
        
    Returns:
        Dynamic range value
    """
    if method == "dr14":
        # DR14 method - RMS difference between segments
        return _measure_dr14(audio, sample_rate)
    elif method == "lra":
        # Loudness Range
        return measure_lufs_range(audio, sample_rate)
    elif method == "crest_factor":
        # Crest factor
        return measure_crest_factor(audio)
    else:
        raise ValueError(f"Unknown dynamic range method: {method}")


def _measure_dr14(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure DR14 dynamic range.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        
    Returns:
        DR14 value in dB
    """
    # Segment audio into 3-second blocks
    block_size = 3 * sample_rate
    blocks = []
    
    for i in range(0, len(audio), block_size):
        block = audio[i:i + block_size]
        if len(block) >= block_size:
            blocks.append(block)
    
    if len(blocks) < 2:
        return 0.0
    
    # Calculate RMS for each block
    rms_values = []
    for block in blocks:
        rms = np.sqrt(np.mean(block**2))
        if rms > 0:
            rms_values.append(20 * np.log10(rms))
    
    if len(rms_values) < 2:
        return 0.0
    
    # DR14 = difference between 95th and 5th percentiles
    rms_values = np.array(rms_values)
    dr14 = np.percentile(rms_values, 95) - np.percentile(rms_values, 5)
    
    return dr14


def measure_punch_impact(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Measure punch and impact characteristics.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with punch metrics
    """
    # Detect transients
    transient_mask = _detect_transients(audio, sample_rate)
    transient_count = np.sum(transient_mask)
    transient_density = transient_count / (len(audio) / sample_rate)
    
    # Measure attack times
    attack_times = _measure_attack_times(audio, sample_rate, transient_mask)
    avg_attack_time = np.mean(attack_times) if attack_times else 0.0
    
    # Measure envelope characteristics
    envelope = _calculate_envelope(audio, sample_rate)
    envelope_variance = np.var(envelope)
    
    # Punch score (combination of factors)
    punch_score = min(100, (transient_density * 10) + (1 / max(0.001, avg_attack_time)) + (envelope_variance * 100))
    
    return {
        "transient_count": transient_count,
        "transient_density": transient_density,
        "avg_attack_time": avg_attack_time,
        "envelope_variance": envelope_variance,
        "punch_score": punch_score
    }


def _detect_transients(audio: np.ndarray, sample_rate: int, threshold: float = 0.3) -> np.ndarray:
    """
    Detect transients in audio signal.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        threshold: Detection threshold
        
    Returns:
        Boolean array indicating transient locations
    """
    # Handle stereo properly - analyze each channel separately
    if len(audio.shape) > 1:
        # Process each channel and combine results
        transient_masks = []
        for ch in range(audio.shape[1]):
            ch_mask = _detect_transients_single_channel(audio[:, ch], sample_rate, threshold)
            transient_masks.append(ch_mask)
        # Combine: transient if ANY channel has one
        return np.logical_or.reduce(transient_masks)
    else:
        return _detect_transients_single_channel(audio, sample_rate, threshold)


def _detect_transients_single_channel(audio: np.ndarray, sample_rate: int, threshold: float = 0.3) -> np.ndarray:
    """
    Detect transients in a single channel.
    """
    # Check if audio is too short for filtering
    if len(audio) < 32:
        return np.zeros(len(audio), dtype=bool)
    
    # High-frequency emphasis - use simple method to avoid filtering issues
    try:
        # Use a very simple high-pass: just first-order difference
        emphasized = np.diff(audio, prepend=audio[0])
    except:
        # Ultimate fallback
        emphasized = audio
    
    # Simple envelope detection
    envelope = np.abs(emphasized)
    
    # Simple smoothing
    window_size = max(3, min(int(0.01 * sample_rate), len(envelope) // 4))
    if window_size >= 3 and len(envelope) >= window_size:
        try:
            smoothed = signal.medfilt(envelope, kernel_size=window_size | 1)
        except:
            smoothed = envelope
    else:
        smoothed = envelope
    
    # Detect rapid increases
    if len(smoothed) > 1:
        diff = np.diff(smoothed)
        if len(diff) > 0:
            std_diff = np.std(diff)
            if std_diff > 0:
                transient_mask = np.zeros(len(audio), dtype=bool)
                transient_mask[1:] = diff > (threshold * std_diff)
            else:
                transient_mask = np.zeros(len(audio), dtype=bool)
        else:
            transient_mask = np.zeros(len(audio), dtype=bool)
    else:
        transient_mask = np.zeros(len(audio), dtype=bool)
    
    return transient_mask


def _measure_attack_times(audio: np.ndarray, sample_rate: int, transient_mask: np.ndarray) -> list:
    """
    Measure attack times for detected transients.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        transient_mask: Boolean array of transient locations
        
    Returns:
        List of attack times in seconds
    """
    attack_times = []
    transient_indices = np.where(transient_mask)[0]
    
    # Convert to mono for attack time analysis
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    for idx in transient_indices:
        # Look back for attack start
        start_idx = max(0, idx - int(0.1 * sample_rate))  # 100ms lookback
        
        # Find where envelope drops below 10% of peak
        segment = audio_mono[start_idx:idx + 1]
        if len(segment) == 0:
            continue
            
        peak_level = np.abs(segment).max()
        threshold_level = 0.1 * peak_level
        
        # Find attack start
        attack_start = start_idx
        for i in range(len(segment) - 1, -1, -1):
            if np.abs(segment[i]) < threshold_level:
                attack_start = start_idx + i
                break
        
        # Calculate attack time
        attack_time = (idx - attack_start) / sample_rate
        if 0.001 <= attack_time <= 0.1:  # Valid range: 1ms to 100ms
            attack_times.append(attack_time)
    
    return attack_times


def _calculate_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculate smooth envelope of audio signal.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        
    Returns:
        Envelope array (preserves stereo if input is stereo)
    """
    # Handle stereo properly - calculate envelope for each channel
    if len(audio.shape) > 1:
        envelopes = []
        for ch in range(audio.shape[1]):
            ch_envelope = _calculate_envelope_single_channel(audio[:, ch], sample_rate)
            envelopes.append(ch_envelope)
        return np.column_stack(envelopes)
    else:
        return _calculate_envelope_single_channel(audio, sample_rate)


def _calculate_envelope_single_channel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculate envelope for a single channel.
    """
    # Check if audio is too short
    if len(audio) < 32:
        return np.abs(audio)
    
    # Simple envelope detection
    envelope = np.abs(audio)
    
    # Simple smoothing with moving average
    window_size = max(3, min(int(0.05 * sample_rate), len(envelope) // 4))
    if window_size >= 3 and len(envelope) >= window_size:
        try:
            smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        except:
            smoothed_envelope = envelope
    else:
        smoothed_envelope = envelope
    
    return smoothed_envelope


def _process_channel_spectrum(args):
    """Process spectrum for a single channel - for parallel processing."""
    channel_audio, sample_rate, fft_size, bands, freqs = args
    
    # Segment audio and average spectrum for this channel
    hop_size = fft_size // 2
    num_segments = (len(channel_audio) - fft_size) // hop_size + 1
    
    spectra = []
    for i in range(num_segments):
        start = i * hop_size
        segment = channel_audio[start:start + fft_size]
        
        if len(segment) < fft_size:
            segment = np.pad(segment, (0, fft_size - len(segment)))
        
        # Apply window
        windowed = segment * signal.windows.hann(fft_size)
        
        # FFT
        spectrum = np.abs(np.fft.fft(windowed))[:fft_size//2]
        spectra.append(spectrum)
    
    # Average spectrum for this channel
    avg_spectrum = np.mean(spectra, axis=0)
    avg_spectrum_db = 20 * np.log10(avg_spectrum + 1e-10)
    
    # Calculate band energies for this channel
    band_energies = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(band_mask):
            band_energy = np.mean(avg_spectrum_db[band_mask])
            band_energies[band_name] = band_energy
        else:
            band_energies[band_name] = -np.inf
    
    # Spectral centroid for this channel
    spectral_centroid = np.sum(freqs * avg_spectrum) / np.sum(avg_spectrum)
    
    # Spectral rolloff for this channel
    cumulative_energy = np.cumsum(avg_spectrum)
    total_energy = cumulative_energy[-1]
    rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    return {
        "band_energies": band_energies,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "avg_spectrum": avg_spectrum_db
    }


def measure_frequency_balance(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Measure frequency balance and spectral characteristics with proper stereo analysis.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with frequency balance metrics including per-channel analysis
    """
    # Calculate FFT
    fft_size = 4096
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate)[:fft_size//2]
    
    # Define frequency bands
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mids": (250, 500),
        "mids": (500, 2000),
        "high_mids": (2000, 4000),
        "presence": (4000, 6000),
        "brilliance": (6000, 20000)
    }
    
    if len(audio.shape) > 1:
        # STEREO ANALYSIS - Process each channel in parallel
        channel_args = []
        for ch in range(audio.shape[1]):
            channel_audio = audio[:, ch]
            channel_args.append((channel_audio, sample_rate, fft_size, bands, freqs))
        
        # Process channels in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            channel_results_list = list(executor.map(_process_channel_spectrum, channel_args))
        
        # Organize results
        channel_results = {}
        for ch, result in enumerate(channel_results_list):
            ch_name = "left" if ch == 0 else "right"
            channel_results[ch_name] = result
        
        # Calculate L/R differences
        lr_differences = {}
        for band_name in bands.keys():
            left_energy = channel_results["left"]["band_energies"][band_name]
            right_energy = channel_results["right"]["band_energies"][band_name]
            if left_energy != -np.inf and right_energy != -np.inf:
                lr_differences[band_name] = left_energy - right_energy
            else:
                lr_differences[band_name] = 0.0
        
        # Combined spectrum (sum of both channels)
        combined_spectrum = (channel_results["left"]["avg_spectrum"] + 
                           channel_results["right"]["avg_spectrum"]) / 2
        
        # Overall balance metrics
        left_bass = channel_results["left"]["band_energies"]["bass"]
        left_mid = channel_results["left"]["band_energies"]["mids"]
        left_treble = channel_results["left"]["band_energies"]["presence"]
        
        return {
            "per_channel": channel_results,
            "lr_differences": lr_differences,
            "combined_spectrum": combined_spectrum,
            "left_bass_mid_balance": left_bass - left_mid,
            "left_mid_treble_balance": left_mid - left_treble,
            "avg_spectrum": combined_spectrum,
            "freqs": freqs
        }
    
    else:
        # MONO ANALYSIS - Single channel
        hop_size = fft_size // 2
        num_segments = (len(audio) - fft_size) // hop_size + 1
        
        spectra = []
        for i in range(num_segments):
            start = i * hop_size
            segment = audio[start:start + fft_size]
            
            if len(segment) < fft_size:
                segment = np.pad(segment, (0, fft_size - len(segment)))
            
            # Apply window
            windowed = segment * signal.windows.hann(fft_size)
            
            # FFT
            spectrum = np.abs(np.fft.fft(windowed))[:fft_size//2]
            spectra.append(spectrum)
        
        # Average spectrum
        avg_spectrum = np.mean(spectra, axis=0)
        avg_spectrum_db = 20 * np.log10(avg_spectrum + 1e-10)
        
        # Calculate band energies
        band_energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_energy = np.mean(avg_spectrum_db[band_mask])
                band_energies[band_name] = band_energy
            else:
                band_energies[band_name] = -np.inf
        
        # Calculate balance metrics
        bass_energy = band_energies["bass"]
        mid_energy = band_energies["mids"]
        treble_energy = band_energies["presence"]
        
        bass_mid_balance = bass_energy - mid_energy
        mid_treble_balance = mid_energy - treble_energy
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * avg_spectrum) / np.sum(avg_spectrum)
        
        # Spectral rolloff
        cumulative_energy = np.cumsum(avg_spectrum)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        return {
            "band_energies": band_energies,
            "bass_mid_balance": bass_mid_balance,
            "mid_treble_balance": mid_treble_balance,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "avg_spectrum": avg_spectrum_db,
            "freqs": freqs
        }


def measure_stereo_characteristics(audio: np.ndarray, sample_rate: int = None) -> dict:
    """
    Measure comprehensive stereo imaging characteristics.
    
    Args:
        audio: Input audio array (must be stereo)
        sample_rate: Sample rate (optional, for frequency-dependent analysis)
        
    Returns:
        Dictionary with stereo metrics
    """
    if len(audio.shape) != 2 or audio.shape[1] != 2:
        return {"error": "Audio must be stereo for stereo analysis"}
    
    left = audio[:, 0]
    right = audio[:, 1]
    
    # Mid/Side conversion
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Measure levels
    left_rms = measure_rms(left)
    right_rms = measure_rms(right)
    mid_rms = measure_rms(mid)
    side_rms = measure_rms(side)
    
    # Stereo width calculation
    if side_rms > -np.inf and mid_rms > -np.inf:
        stereo_width = side_rms - mid_rms
    else:
        stereo_width = 0.0
    
    # Correlation coefficient
    correlation = np.corrcoef(left, right)[0, 1]
    
    # Phase coherence
    phase_coherence = np.abs(correlation)
    
    # Balance (L/R difference)
    lr_balance = left_rms - right_rms
    
    # Mono compatibility (what happens when summed to mono)
    mono_sum = (left + right) / 2
    mono_rms = measure_rms(mono_sum)
    
    # M/S content analysis
    ms_analysis = _analyze_ms_content(mid, side)
    
    # Phase analysis
    phase_analysis = _analyze_phase_characteristics(left, right, sample_rate)
    
    # Frequency-dependent stereo width (if sample rate provided)
    freq_dependent_width = {}
    if sample_rate is not None:
        freq_dependent_width = _analyze_frequency_dependent_stereo_width(left, right, sample_rate)
    
    return {
        "left_rms": left_rms,
        "right_rms": right_rms,
        "mid_rms": mid_rms,
        "side_rms": side_rms,
        "stereo_width": stereo_width,
        "correlation": correlation,
        "phase_coherence": phase_coherence,
        "lr_balance": lr_balance,
        "mono_rms": mono_rms,
        "ms_analysis": ms_analysis,
        "phase_analysis": phase_analysis,
        "frequency_dependent_width": freq_dependent_width
    }


def _analyze_ms_content(mid: np.ndarray, side: np.ndarray) -> dict:
    """
    Analyze Mid/Side content distribution and energy.
    
    Args:
        mid: Mid channel signal
        side: Side channel signal
        
    Returns:
        Dictionary with M/S analysis metrics
    """
    # Energy analysis
    mid_energy = np.sum(mid**2)
    side_energy = np.sum(side**2)
    total_energy = mid_energy + side_energy
    
    # Energy distribution
    mid_energy_percent = (mid_energy / total_energy * 100) if total_energy > 0 else 0
    side_energy_percent = (side_energy / total_energy * 100) if total_energy > 0 else 0
    
    # RMS levels
    mid_rms_linear = np.sqrt(np.mean(mid**2))
    side_rms_linear = np.sqrt(np.mean(side**2))
    
    # M/S ratio
    ms_ratio = (side_rms_linear / mid_rms_linear) if mid_rms_linear > 0 else 0
    
    # Dynamic range in M/S
    mid_peak = np.abs(mid).max()
    side_peak = np.abs(side).max()
    
    mid_crest = (20 * np.log10(mid_peak / mid_rms_linear)) if mid_rms_linear > 0 else 0
    side_crest = (20 * np.log10(side_peak / side_rms_linear)) if side_rms_linear > 0 else 0
    
    # Stereo content classification
    if ms_ratio < 0.1:
        stereo_content = "Mono-like"
    elif ms_ratio < 0.5:
        stereo_content = "Narrow"
    elif ms_ratio < 1.0:
        stereo_content = "Moderate"
    elif ms_ratio < 2.0:
        stereo_content = "Wide"
    else:
        stereo_content = "Very Wide"
    
    return {
        "mid_energy_percent": mid_energy_percent,
        "side_energy_percent": side_energy_percent,
        "ms_ratio": ms_ratio,
        "mid_crest_factor": mid_crest,
        "side_crest_factor": side_crest,
        "stereo_content_classification": stereo_content,
        "mid_rms_linear": mid_rms_linear,
        "side_rms_linear": side_rms_linear
    }


def _analyze_phase_characteristics(left: np.ndarray, right: np.ndarray, sample_rate: int = None) -> dict:
    """
    Analyze phase characteristics between left and right channels.
    
    Args:
        left: Left channel signal
        right: Right channel signal
        sample_rate: Sample rate (optional, for frequency-dependent analysis)
        
    Returns:
        Dictionary with phase analysis metrics
    """
    # Cross-correlation for phase relationship
    cross_corr = signal.correlate(left, right, mode='full')
    lags = signal.correlation_lags(len(left), len(right), mode='full')
    
    # Find peak correlation and its lag
    peak_idx = np.argmax(np.abs(cross_corr))
    peak_lag = lags[peak_idx]
    peak_correlation = cross_corr[peak_idx]
    
    # Phase delay in samples and time
    phase_delay_samples = peak_lag
    phase_delay_ms = (phase_delay_samples / sample_rate * 1000) if sample_rate else 0
    
    # Polarity check
    if peak_correlation < 0:
        polarity_status = "Inverted"
    else:
        polarity_status = "Normal"
    
    # Phase coherence analysis
    coherence_analysis = {}
    if sample_rate:
        coherence_analysis = _analyze_phase_coherence_by_frequency(left, right, sample_rate)
    
    # Goniometer-style analysis
    gonio_analysis = _analyze_goniometer_characteristics(left, right)
    
    return {
        "phase_delay_samples": phase_delay_samples,
        "phase_delay_ms": phase_delay_ms,
        "peak_correlation": float(peak_correlation),
        "polarity_status": polarity_status,
        "coherence_by_frequency": coherence_analysis,
        "goniometer_analysis": gonio_analysis
    }


def _analyze_phase_coherence_by_frequency(left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict:
    """
    Analyze phase coherence across frequency bands.
    
    Args:
        left: Left channel signal
        right: Right channel signal
        sample_rate: Sample rate
        
    Returns:
        Dictionary with frequency-dependent phase coherence
    """
    # Define frequency bands
    bands = {
        "bass": (60, 250),
        "low_mids": (250, 500),
        "mids": (500, 2000),
        "high_mids": (2000, 4000),
        "highs": (4000, 8000)
    }
    
    # Filter signals and measure coherence per band
    coherence_by_band = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        try:
            # Bandpass filter
            nyquist = sample_rate / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.99)
            
            if low_norm < 0.99 and high_norm > low_norm:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                
                left_filtered = signal.filtfilt(b, a, left)
                right_filtered = signal.filtfilt(b, a, right)
                
                # Correlation in this band
                correlation = np.corrcoef(left_filtered, right_filtered)[0, 1]
                coherence_by_band[band_name] = float(correlation)
            else:
                coherence_by_band[band_name] = 0.0
                
        except Exception:
            coherence_by_band[band_name] = 0.0
    
    return coherence_by_band


def _analyze_goniometer_characteristics(left: np.ndarray, right: np.ndarray) -> dict:
    """
    Analyze goniometer/vectorscope characteristics.
    
    Args:
        left: Left channel signal
        right: Right channel signal
        
    Returns:
        Dictionary with goniometer analysis
    """
    # Downsample for analysis
    downsample_factor = max(1, len(left) // 50000)
    left_ds = left[::downsample_factor]
    right_ds = right[::downsample_factor]
    
    # Calculate angles (phase relationships)
    angles = np.arctan2(right_ds, left_ds)
    
    # Angle distribution analysis
    angle_std = np.std(angles)
    angle_range = np.ptp(angles)
    
    # Quadrant analysis
    q1 = np.sum((left_ds > 0) & (right_ds > 0))  # Both positive
    q2 = np.sum((left_ds < 0) & (right_ds > 0))  # Left neg, Right pos
    q3 = np.sum((left_ds < 0) & (right_ds < 0))  # Both negative
    q4 = np.sum((left_ds > 0) & (right_ds < 0))  # Left pos, Right neg
    
    total_samples = len(left_ds)
    quadrant_distribution = {
        "q1_percent": (q1 / total_samples * 100) if total_samples > 0 else 0,
        "q2_percent": (q2 / total_samples * 100) if total_samples > 0 else 0,
        "q3_percent": (q3 / total_samples * 100) if total_samples > 0 else 0,
        "q4_percent": (q4 / total_samples * 100) if total_samples > 0 else 0
    }
    
    # Stereo field width estimation
    radius = np.sqrt(left_ds**2 + right_ds**2)
    avg_radius = np.mean(radius)
    
    return {
        "angle_std": float(angle_std),
        "angle_range": float(angle_range),
        "quadrant_distribution": quadrant_distribution,
        "avg_radius": float(avg_radius),
        "phase_spread": float(angle_std * 180 / np.pi)  # Convert to degrees
    }


def _analyze_frequency_dependent_stereo_width(left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict:
    """
    Analyze stereo width across frequency bands.
    
    Args:
        left: Left channel signal
        right: Right channel signal
        sample_rate: Sample rate
        
    Returns:
        Dictionary with frequency-dependent stereo width
    """
    # Define frequency bands
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mids": (250, 500),
        "mids": (500, 2000),
        "high_mids": (2000, 4000),
        "highs": (4000, 8000),
        "air": (8000, 20000)
    }
    
    width_by_band = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        try:
            # Bandpass filter
            nyquist = sample_rate / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.99)
            
            if low_norm < 0.99 and high_norm > low_norm:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                
                left_filtered = signal.filtfilt(b, a, left)
                right_filtered = signal.filtfilt(b, a, right)
                
                # Calculate M/S for this band
                mid_band = (left_filtered + right_filtered) / 2
                side_band = (left_filtered - right_filtered) / 2
                
                # RMS levels
                mid_rms = np.sqrt(np.mean(mid_band**2))
                side_rms = np.sqrt(np.mean(side_band**2))
                
                # Stereo width for this band
                if mid_rms > 0:
                    width_db = 20 * np.log10(side_rms / mid_rms)
                else:
                    width_db = -np.inf
                
                width_by_band[band_name] = float(width_db)
            else:
                width_by_band[band_name] = -np.inf
                
        except Exception:
            width_by_band[band_name] = -np.inf
    
    return width_by_band


def _analyze_basic_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Basic audio metrics analysis - for parallel processing."""
    peak_db = measure_peak_level(audio)
    rms_db = measure_rms(audio)
    crest_factor = measure_crest_factor(audio)
    
    return {
        "peak_db": peak_db,
        "rms_db": rms_db,
        "crest_factor": crest_factor,
        "sample_rate": sample_rate,
        "duration": len(audio) / sample_rate,
        "channels": audio.shape[1] if len(audio.shape) > 1 else 1
    }


def _analyze_loudness_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Loudness metrics analysis - for parallel processing."""
    lufs = measure_lufs(audio, sample_rate)
    lra = measure_lufs_range(audio, sample_rate)
    true_peak_db = measure_true_peak_level(audio, sample_rate)
    
    return {
        "lufs": lufs,
        "lra": lra,
        "true_peak_db": true_peak_db
    }


def _analyze_dynamic_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Dynamic range metrics analysis - for parallel processing."""
    dr14 = measure_dynamic_range(audio, sample_rate, "dr14")
    punch_metrics = measure_punch_impact(audio, sample_rate)
    
    return {
        "dr14": dr14,
        "punch_metrics": punch_metrics
    }


def _analyze_statistics(audio: np.ndarray) -> dict:
    """Statistical analysis - for parallel processing."""
    audio_flat = audio.flatten()
    return {
        "mean": np.mean(audio_flat),
        "std": np.std(audio_flat),
        "skewness": skew(audio_flat),
        "kurtosis": kurtosis(audio_flat),
        "zero_crossings": np.sum(np.diff(np.signbit(audio_flat)))
    }


def comprehensive_analysis(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Perform comprehensive audio analysis with parallel processing.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with all analysis results
    """
    debug("Starting comprehensive audio analysis with parallel processing...")
    
    # Create analysis tasks for parallel execution
    analysis_tasks = [
        ("basic_metrics", partial(_analyze_basic_metrics, audio, sample_rate)),
        ("loudness_metrics", partial(_analyze_loudness_metrics, audio, sample_rate)),
        ("dynamic_metrics", partial(_analyze_dynamic_metrics, audio, sample_rate)),
        ("frequency_metrics", partial(measure_frequency_balance, audio, sample_rate)),
        ("statistics", partial(_analyze_statistics, audio))
    ]
    
    # Execute analysis tasks in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_name = {executor.submit(task): name for name, task in analysis_tasks}
        
        for future in future_to_name:
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                debug(f"Error in {name} analysis: {e}")
                results[name] = {}
    
    # Stereo characteristics (if stereo) - run separately as it's quick
    stereo_metrics = {}
    if len(audio.shape) == 2 and audio.shape[1] == 2:
        stereo_metrics = measure_stereo_characteristics(audio, sample_rate)
    
    # Compliance checks - run separately as it depends on loudness metrics
    atmos_compliance = check_atmos_compliance(audio, sample_rate)
    
    # Combine all results
    analysis_result = {
        "basic_metrics": results.get("basic_metrics", {}),
        "loudness_metrics": results.get("loudness_metrics", {}),
        "dynamic_metrics": results.get("dynamic_metrics", {}),
        "frequency_metrics": results.get("frequency_metrics", {}),
        "stereo_metrics": stereo_metrics,
        "compliance": atmos_compliance,
        "statistics": results.get("statistics", {})
    }
    
    debug("Comprehensive audio analysis completed")
    return analysis_result