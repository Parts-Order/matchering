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
import copy
from .. import Config
from ..log import debug
from ..oversampling import upsample, downsample, get_oversampling_factor, is_oversampling_needed
from ..true_peak import measure_true_peak_level, is_true_peak_compliant
from .hyrax import limit as hyrax_limit
from .envelope import enhanced_envelope_processing
from ..multiband import compress_multiband, get_preset_by_name


def limit_with_advanced_envelope(audio: np.ndarray, config: Config, mode: str = "transparent", use_filtering: bool = True) -> np.ndarray:
    """
    Apply limiting with advanced envelope shaping instead of standard Hyrax.
    
    Args:
        audio: Input audio array
        config: Matchering configuration
        mode: Limiting mode
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Limited audio with advanced envelope characteristics
    """
    from ..dsp import rectify, flip
    
    # Use the same rectification as original Hyrax
    rectified = rectify(audio, config.threshold)
    
    if np.all(np.isclose(rectified, 1.0)):
        debug("No limiting needed with advanced envelope")
        return audio
    
    # Calculate gain envelope with advanced shaping
    gain_hard_clip = flip(1.0 / rectified)
    
    # Apply enhanced envelope processing instead of standard attack/release
    gain_envelope = enhanced_envelope_processing(gain_hard_clip, audio, config, mode, use_filtering)
    
    # Apply final gain
    gain_final = flip(gain_envelope)
    
    return audio * gain_final[:, None]


def limit_with_oversampling(audio: np.ndarray, config: Config, enable_true_peak: bool = True, lookahead_ms: float = 5.0, limiting_mode: str = "transparent", multiband_preset: str = None, multiband_enabled: bool = False, use_filtering: bool = True) -> np.ndarray:
    """
    Enhanced Hyrax limiter with multiband compression, oversampling, lookahead, and advanced envelope shaping.
    
    Args:
        audio: Input audio array (samples, channels)
        config: Matchering configuration
        enable_true_peak: Enable true peak limiting with oversampling
        lookahead_ms: Lookahead buffer in milliseconds (0-20ms)
        limiting_mode: Limiting mode ("transparent", "punchy", "aggressive")
        multiband_preset: Multiband compressor preset ("default", "3band", "broadcast", "mastering")
        multiband_enabled: Enable multiband compression before limiting
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Limited audio with multiband compression, true peak compliance and advanced envelope shaping
    """
    debug(f"Enhanced limiter started with {lookahead_ms}ms lookahead and oversampling support")
    
    # Apply multiband compression first if enabled
    processing_audio = audio
    if multiband_enabled and multiband_preset:
        debug(f"Applying multiband compression with '{multiband_preset}' preset")
        processing_audio = compress_multiband(audio, config.internal_sample_rate, multiband_preset)
    
    # Apply lookahead delay if specified
    if lookahead_ms > 0:
        lookahead_samples = int(lookahead_ms * config.internal_sample_rate / 1000)
        debug(f"Applying {lookahead_samples} sample lookahead buffer")
        
        # Pad audio with lookahead buffer
        if len(processing_audio.shape) == 1:
            audio_with_lookahead = np.pad(processing_audio, (lookahead_samples, 0), mode='constant')
        else:
            audio_with_lookahead = np.pad(processing_audio, ((lookahead_samples, 0), (0, 0)), mode='constant')
    else:
        audio_with_lookahead = processing_audio
        lookahead_samples = 0
    
    # Check if oversampling is needed
    if not enable_true_peak or not is_oversampling_needed(config.internal_sample_rate, enable_true_peak):
        filter_method = "filtering" if use_filtering else "mathematical"
        debug(f"Using advanced envelope limiter in '{limiting_mode}' mode (no oversampling, {filter_method} envelope)")
        limited = limit_with_advanced_envelope(audio_with_lookahead, config, limiting_mode, use_filtering)
        
        # Remove lookahead delay from output
        if lookahead_samples > 0:
            limited = limited[lookahead_samples:]
        
        return limited
    
    # Measure original true peak level
    original_peak_db = measure_true_peak_level(audio, config.internal_sample_rate)
    debug(f"Original true peak level: {original_peak_db:.2f} dBFS")
    
    # Determine oversampling factor
    factor = get_oversampling_factor(config.internal_sample_rate)
    debug(f"Using {factor}x oversampling")
    
    # Upsample audio with lookahead
    debug("Upsampling audio with lookahead...")
    upsampled_audio = upsample(audio_with_lookahead, config.internal_sample_rate, factor)
    
    # Create oversampled configuration
    oversampled_config = copy.deepcopy(config)
    oversampled_config.internal_sample_rate *= factor
    
    # Apply advanced envelope limiting at oversampled rate
    filter_method = "filtering" if use_filtering else "mathematical"
    debug(f"Applying advanced envelope limiting in '{limiting_mode}' mode at oversampled rate ({filter_method} envelope)...")
    limited_upsampled = limit_with_advanced_envelope(upsampled_audio, oversampled_config, limiting_mode, use_filtering)
    
    # Downsample back to original rate
    debug("Downsampling limited audio...")
    limited_downsampled = downsample(limited_upsampled, oversampled_config.internal_sample_rate, factor)
    
    # Remove lookahead delay from output
    if lookahead_samples > 0:
        limited_audio = limited_downsampled[lookahead_samples:]
    else:
        limited_audio = limited_downsampled
    
    # Verify true peak level after limiting
    final_peak_db = measure_true_peak_level(limited_audio, config.internal_sample_rate)
    debug(f"Final true peak level: {final_peak_db:.2f} dBFS")
    
    # Check broadcast compliance
    ebu_compliant = is_true_peak_compliant(limited_audio, config.internal_sample_rate, "EBU")
    debug(f"EBU R128 true peak compliant: {ebu_compliant}")
    
    debug(f"Enhanced limiter completed with {lookahead_ms}ms lookahead")
    return limited_audio


def limit_with_external_sidechain_oversampling(
    input_audio: np.ndarray, 
    sidechain_audio: np.ndarray, 
    config: Config, 
    enable_true_peak: bool = True,
    lookahead_ms: float = 5.0,
    limiting_mode: str = "transparent",
    multiband_preset: str = None,
    multiband_enabled: bool = False,
    use_filtering: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced external sidechain limiter with multiband compression, oversampling, lookahead, and advanced envelope shaping.
    
    Args:
        input_audio: Audio to be limited
        sidechain_audio: Audio that triggers limiting
        config: Matchering configuration
        enable_true_peak: Enable true peak limiting with oversampling
        lookahead_ms: Lookahead buffer in milliseconds (0-20ms)
        limiting_mode: Limiting mode ("transparent", "punchy", "aggressive")
        multiband_preset: Multiband compressor preset ("default", "3band", "broadcast", "mastering")
        multiband_enabled: Enable multiband compression before limiting
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Tuple of (limited_audio, gain_envelope)
    """
    debug(f"Enhanced external sidechain limiter started with {lookahead_ms}ms lookahead")
    
    # Apply multiband compression to input audio first if enabled
    processing_input = input_audio
    if multiband_enabled and multiband_preset:
        debug(f"Applying multiband compression to input with '{multiband_preset}' preset")
        processing_input = compress_multiband(input_audio, config.internal_sample_rate, multiband_preset)
    
    # Apply lookahead delay if specified
    if lookahead_ms > 0:
        lookahead_samples = int(lookahead_ms * config.internal_sample_rate / 1000)
        debug(f"Applying {lookahead_samples} sample lookahead buffer")
        
        # Pad both input and sidechain with lookahead buffer
        if len(processing_input.shape) == 1:
            input_with_lookahead = np.pad(processing_input, (lookahead_samples, 0), mode='constant')
            sidechain_with_lookahead = np.pad(sidechain_audio, (lookahead_samples, 0), mode='constant')
        else:
            input_with_lookahead = np.pad(processing_input, ((lookahead_samples, 0), (0, 0)), mode='constant')
            sidechain_with_lookahead = np.pad(sidechain_audio, ((lookahead_samples, 0), (0, 0)), mode='constant')
    else:
        input_with_lookahead = processing_input
        sidechain_with_lookahead = sidechain_audio
        lookahead_samples = 0
    
    # Import internal functions from hyrax
    import math
    from scipy import signal
    from scipy.ndimage import maximum_filter1d
    from ..dsp import rectify, flip, max_mix
    from ..utils import make_odd, ms_to_samples
    
    def __sliding_window_fast(array: np.ndarray, window_size: int, mode: str = "attack") -> np.ndarray:
        if mode == "attack":
            window_size = make_odd(window_size)
            return maximum_filter1d(array, size=(2 * window_size - 1))
        half_window_size = (window_size - 1) // 2
        array = np.pad(array, (half_window_size, 0))
        return maximum_filter1d(array, size=window_size)[:-half_window_size]

    def __process_attack(array: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
        attack = ms_to_samples(config.limiter.attack, config.internal_sample_rate)
        slided_input = __sliding_window_fast(array, attack, mode="attack")
        coef = math.exp(config.limiter.attack_filter_coefficient / attack)
        b = [1 - coef]
        a = [1, -coef]
        output = signal.filtfilt(b, a, slided_input)
        return output, slided_input

    def __process_release(array: np.ndarray, config: Config) -> np.ndarray:
        hold = ms_to_samples(config.limiter.hold, config.internal_sample_rate)
        slided_input = __sliding_window_fast(array, hold, mode="hold")
        b, a = signal.butter(
            config.limiter.hold_filter_order,
            config.limiter.hold_filter_coefficient,
            fs=config.internal_sample_rate,
        )
        hold_output = signal.lfilter(b, a, slided_input)
        b, a = signal.butter(
            config.limiter.release_filter_order,
            config.limiter.release_filter_coefficient / config.limiter.release,
            fs=config.internal_sample_rate,
        )
        release_output = signal.lfilter(b, a, np.maximum(slided_input, hold_output))
        return np.maximum(hold_output, release_output)
    
    # Check if oversampling is needed
    if not enable_true_peak or not is_oversampling_needed(config.internal_sample_rate, enable_true_peak):
        debug("Using standard external sidechain limiting (no oversampling)")
        
        # Standard processing without oversampling
        rectified = rectify(sidechain_with_lookahead, config.threshold)
        
        if np.all(np.isclose(rectified, 1.0)):
            gain_envelope = np.ones(input_with_lookahead.shape[0])
            limited = input_with_lookahead
        else:
            gain_hard_clip = flip(1.0 / rectified)
            gain_attack, gain_hard_clip_slided = __process_attack(np.copy(gain_hard_clip), config)
            gain_release = __process_release(np.copy(gain_hard_clip_slided), config)
            gain_envelope = flip(max_mix(gain_hard_clip, gain_attack, gain_release))
            gain_envelope = np.clip(gain_envelope, 0.0, 1.0)
            
            limited = input_with_lookahead * gain_envelope[:, None]
        
        # Remove lookahead delay from outputs
        if lookahead_samples > 0:
            limited_audio = limited[lookahead_samples:]
            final_gain = gain_envelope[lookahead_samples:]
        else:
            limited_audio = limited
            final_gain = gain_envelope
            
        return limited_audio, final_gain
    
    # Oversampled processing
    debug("Using oversampled external sidechain limiting")
    
    # Determine oversampling factor
    factor = get_oversampling_factor(config.internal_sample_rate)
    debug(f"Using {factor}x oversampling")
    
    # Upsample both input and sidechain with lookahead
    debug("Upsampling input and sidechain audio with lookahead...")
    upsampled_input = upsample(input_with_lookahead, config.internal_sample_rate, factor)
    upsampled_sidechain = upsample(sidechain_with_lookahead, config.internal_sample_rate, factor)
    
    # Create oversampled configuration
    oversampled_config = copy.deepcopy(config)
    oversampled_config.internal_sample_rate *= factor
    
    # Process at oversampled rate
    rectified = rectify(upsampled_sidechain, oversampled_config.threshold)
    
    if np.all(np.isclose(rectified, 1.0)):
        debug("No limiting needed at oversampled rate")
        gain_envelope = np.ones(upsampled_input.shape[0])
        limited_upsampled = upsampled_input
    else:
        # Calculate gain envelope from oversampled sidechain
        gain_hard_clip = flip(1.0 / rectified)
        gain_attack, gain_hard_clip_slided = __process_attack(np.copy(gain_hard_clip), oversampled_config)
        gain_release = __process_release(np.copy(gain_hard_clip_slided), oversampled_config)
        gain_envelope = flip(max_mix(gain_hard_clip, gain_attack, gain_release))
        gain_envelope = np.clip(gain_envelope, 0.0, 1.0)
        
        # Apply gain envelope to upsampled input
        limited_upsampled = upsampled_input * gain_envelope[:, None]
    
    # Downsample results
    debug("Downsampling limited audio and gain envelope...")
    limited_downsampled = downsample(limited_upsampled, oversampled_config.internal_sample_rate, factor)
    
    # Downsample gain envelope (use average pooling for smoother result)
    downsampled_length = input_with_lookahead.shape[0]
    downsampled_gain = np.zeros(downsampled_length)
    
    for i in range(downsampled_length):
        start_idx = i * factor
        end_idx = min((i + 1) * factor, len(gain_envelope))
        downsampled_gain[i] = np.mean(gain_envelope[start_idx:end_idx])
    
    # Remove lookahead delay from outputs
    if lookahead_samples > 0:
        limited_audio = limited_downsampled[lookahead_samples:]
        final_gain = downsampled_gain[lookahead_samples:]
    else:
        limited_audio = limited_downsampled
        final_gain = downsampled_gain
    
    # Verify true peak compliance
    final_peak_db = measure_true_peak_level(limited_audio, config.internal_sample_rate)
    debug(f"Final true peak level: {final_peak_db:.2f} dBFS")
    
    debug(f"Enhanced external sidechain limiter completed with {lookahead_ms}ms lookahead")
    return limited_audio, final_gain


def get_limiter_analysis(audio: np.ndarray, config: Config) -> dict:
    """
    Analyze audio and provide limiter recommendations.
    
    Args:
        audio: Input audio array
        config: Matchering configuration
        
    Returns:
        Dictionary with analysis results and recommendations
    """
    debug("Analyzing audio for limiter recommendations")
    
    # Measure characteristics
    peak_db = 20 * np.log10(np.abs(audio).max()) if np.abs(audio).max() > 0 else -np.inf
    true_peak_db = measure_true_peak_level(audio, config.internal_sample_rate)
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2))) if np.sqrt(np.mean(audio**2)) > 0 else -np.inf
    crest_factor = peak_db - rms_db if rms_db > -np.inf else np.inf
    
    # Check compliance
    ebu_compliant = is_true_peak_compliant(audio, config.internal_sample_rate, "EBU")
    atsc_compliant = is_true_peak_compliant(audio, config.internal_sample_rate, "ATSC")
    
    # Recommendations
    needs_limiting = true_peak_db > -0.1
    needs_oversampling = is_oversampling_needed(config.internal_sample_rate, True)
    intersample_risk = (true_peak_db - peak_db) > 0.1
    
    analysis = {
        "peak_db": peak_db,
        "true_peak_db": true_peak_db,
        "rms_db": rms_db,
        "crest_factor": crest_factor,
        "intersample_difference": true_peak_db - peak_db,
        "ebu_compliant": ebu_compliant,
        "atsc_compliant": atsc_compliant,
        "needs_limiting": needs_limiting,
        "needs_oversampling": needs_oversampling,
        "intersample_risk": intersample_risk,
        "recommended_threshold": min(-0.1, true_peak_db - 0.1),
        "sample_rate": config.internal_sample_rate,
        "oversampling_factor": get_oversampling_factor(config.internal_sample_rate)
    }
    
    debug(f"Analysis complete: peak={peak_db:.2f}dB, true_peak={true_peak_db:.2f}dB, "
          f"crest={crest_factor:.1f}dB, needs_limiting={needs_limiting}")
    
    return analysis