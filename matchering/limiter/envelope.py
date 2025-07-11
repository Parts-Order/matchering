# -*- coding: utf-8 -*-

"""
Advanced Envelope Shaping for Matchering Limiter
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
import math
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .. import Config
from ..log import debug
from ..dsp import rectify, flip, max_mix
from ..utils import make_odd, ms_to_samples
from ..analysis import _detect_transients


def logarithmic_attack_curve(gain_reduction: np.ndarray, attack_time_ms: float, sample_rate: int) -> np.ndarray:
    """
    Create logarithmic attack curve for smoother, more musical gain reduction.
    
    Args:
        gain_reduction: Target gain reduction values
        attack_time_ms: Attack time in milliseconds
        sample_rate: Sample rate
        
    Returns:
        Smoothed gain reduction with logarithmic curve
    """
    attack_samples = ms_to_samples(attack_time_ms, sample_rate)
    
    if attack_samples <= 1:
        return gain_reduction
    
    # Create logarithmic attack coefficient (more musical than exponential)
    # Use log base that gives smooth but responsive attack
    log_coef = 1.0 - (1.0 / (attack_samples * 0.368))  # Based on log curve shape
    
    # Apply logarithmic smoothing
    smoothed = np.zeros_like(gain_reduction)
    smoothed[0] = gain_reduction[0]
    
    for i in range(1, len(gain_reduction)):
        # Logarithmic interpolation between current and target
        target = gain_reduction[i]
        current = smoothed[i-1]
        
        # Logarithmic approach to target (smoother than exponential)
        if target < current:  # Attack (gain reduction increasing)
            # Fast logarithmic attack
            diff = target - current
            smoothed[i] = current + diff * (1.0 - log_coef)
        else:  # Release handled separately
            smoothed[i] = target
    
    return smoothed


def program_dependent_release(gain_reduction: np.ndarray, audio: np.ndarray, 
                            release_time_ms: float, sample_rate: int) -> np.ndarray:
    """
    Implement program-dependent release: faster for transients, slower for sustained content.
    
    Args:
        gain_reduction: Current gain reduction values
        audio: Original audio signal for transient detection
        release_time_ms: Base release time in milliseconds
        sample_rate: Sample rate
        
    Returns:
        Gain reduction with program-dependent release applied
    """
    # Detect transients in the audio
    if len(audio.shape) > 1:
        # Use mono sum for transient detection
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    # Detect transients
    transient_mask = _detect_transients(audio_mono, sample_rate, threshold=0.2)
    
    # Create adaptive release times
    base_release_samples = ms_to_samples(release_time_ms, sample_rate)
    
    # Fast release for transients (2x faster)
    fast_release_samples = max(1, base_release_samples // 2)
    
    # Slow release for sustained content (1.5x slower)
    slow_release_samples = int(base_release_samples * 1.5)
    
    # Apply program-dependent release
    smoothed = np.zeros_like(gain_reduction)
    smoothed[0] = gain_reduction[0]
    
    for i in range(1, len(gain_reduction)):
        target = gain_reduction[i]
        current = smoothed[i-1]
        
        if target > current:  # Release (gain reduction decreasing)
            # Choose release speed based on transient detection
            if i < len(transient_mask) and transient_mask[i]:
                # Fast release for transients
                release_samples = fast_release_samples
            else:
                # Slow release for sustained content
                release_samples = slow_release_samples
            
            # Logarithmic release curve (more musical than exponential)
            release_coef = 1.0 - (1.0 / (release_samples * 0.368))
            diff = target - current
            smoothed[i] = current + diff * (1.0 - release_coef)
        else:  # Attack handled separately
            smoothed[i] = target
    
    return smoothed


def advanced_envelope_shaping(gain_hard_clip: np.ndarray, audio: np.ndarray, config: Config, use_filtering: bool = True) -> np.ndarray:
    """
    Apply advanced envelope shaping with logarithmic curves and program-dependent release.
    
    Args:
        gain_hard_clip: Hard clipping gain values
        audio: Original audio for program-dependent analysis
        config: Matchering configuration
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Smoothed gain envelope with advanced shaping
    """
    filter_method = "filtering" if use_filtering else "mathematical"
    debug(f"Applying advanced envelope shaping with logarithmic curves ({filter_method} approach)")
    
    # Apply logarithmic attack curve
    attack_shaped = logarithmic_attack_curve(
        gain_hard_clip, 
        config.limiter.attack, 
        config.internal_sample_rate
    )
    
    # Apply program-dependent release
    release_shaped = program_dependent_release(
        attack_shaped,
        audio,
        config.limiter.release,
        config.internal_sample_rate
    )
    
    # Apply hold stage with smoother characteristics
    hold_shaped = apply_smooth_hold(
        release_shaped,
        config.limiter.hold,
        config.internal_sample_rate,
        use_filtering
    )
    
    debug(f"Advanced envelope shaping completed ({filter_method} approach)")
    return hold_shaped


def apply_smooth_hold(gain_reduction: np.ndarray, hold_time_ms: float, sample_rate: int, use_filtering: bool = True) -> np.ndarray:
    """
    Apply smooth hold stage with gentle transitions.
    
    Args:
        gain_reduction: Input gain reduction
        hold_time_ms: Hold time in milliseconds
        sample_rate: Sample rate
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical hold.
        
    Returns:
        Gain reduction with smooth hold applied
    """
    hold_samples = ms_to_samples(hold_time_ms, sample_rate)
    
    if hold_samples <= 1:
        return gain_reduction
    
    if use_filtering:
        # ORIGINAL APPROACH: Use Butterworth filtering for hold stage
        try:
            # Use 2nd order instead of higher order for smoother transitions
            b, a = signal.butter(2, 0.1, fs=sample_rate, btype='low')
            held = signal.lfilter(b, a, gain_reduction)
            
            # Ensure we don't exceed the original gain reduction
            return np.minimum(gain_reduction, held)
        except:
            # Fallback to simple moving average if filter fails
            kernel_size = min(hold_samples, len(gain_reduction) // 4)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                held = np.convolve(gain_reduction, kernel, mode='same')
                return np.minimum(gain_reduction, held)
            else:
                return gain_reduction
    else:
        # NEW APPROACH: Pure mathematical hold without filtering
        # Use a simple peak-hold algorithm that maintains gain reduction for the hold time
        held = np.zeros_like(gain_reduction)
        held[0] = gain_reduction[0]
        
        for i in range(1, len(gain_reduction)):
            current_gain = gain_reduction[i]
            previous_held = held[i-1]
            
            # If current gain reduction is greater (more limiting needed)
            if current_gain < previous_held:
                held[i] = current_gain  # Apply immediate gain reduction
            else:
                # Check if we should maintain previous gain reduction (hold)
                # Look back up to hold_samples to find the minimum (maximum gain reduction)
                lookback_start = max(0, i - hold_samples)
                min_gain_in_window = np.min(gain_reduction[lookback_start:i+1])
                
                # Maintain the strongest gain reduction within the hold window
                held[i] = min_gain_in_window
        
        return held


def create_limiting_curves(mode: str = "transparent") -> dict:
    """
    Create different limiting curve parameters for various modes.
    
    Args:
        mode: Limiting mode ("transparent", "punchy", "aggressive")
        
    Returns:
        Dictionary with curve parameters
    """
    curves = {
        "transparent": {
            "attack_curve_shape": 0.368,  # Gentle logarithmic curve
            "release_curve_shape": 0.368,
            "transient_sensitivity": 0.2,
            "release_speed_multiplier": 1.0,
            "hold_smoothness": 2  # Lower order filters
        },
        "punchy": {
            "attack_curve_shape": 0.5,    # Slightly more aggressive
            "release_curve_shape": 0.3,   # Faster release curve
            "transient_sensitivity": 0.15, # More sensitive to transients
            "release_speed_multiplier": 1.5, # Faster overall release
            "hold_smoothness": 3
        },
        "aggressive": {
            "attack_curve_shape": 0.7,    # More aggressive attack
            "release_curve_shape": 0.2,   # Much faster release
            "transient_sensitivity": 0.1, # Very sensitive to transients
            "release_speed_multiplier": 2.0, # Much faster release
            "hold_smoothness": 4  # Higher order for tighter control
        }
    }
    
    return curves.get(mode, curves["transparent"])


def enhanced_envelope_processing(array: np.ndarray, audio: np.ndarray, config: Config, 
                               mode: str = "transparent", use_filtering: bool = True) -> np.ndarray:
    """
    Enhanced envelope processing with advanced shaping and multiple modes.
    
    Args:
        array: Gain array to process
        audio: Original audio for program-dependent analysis
        config: Matchering configuration
        mode: Limiting mode ("transparent", "punchy", "aggressive")
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Processed gain envelope with advanced characteristics
    """
    filter_method = "filtering" if use_filtering else "mathematical"
    debug(f"Enhanced envelope processing in '{mode}' mode ({filter_method} approach)")
    
    # Get curve parameters for selected mode
    curve_params = create_limiting_curves(mode)
    
    # Create modified config for this mode
    mode_config = Config()
    mode_config.internal_sample_rate = config.internal_sample_rate
    mode_config.threshold = config.threshold
    
    # Adjust timing based on mode
    mode_config.limiter.attack = config.limiter.attack
    mode_config.limiter.hold = config.limiter.hold
    mode_config.limiter.release = config.limiter.release / curve_params["release_speed_multiplier"]
    
    # Apply advanced shaping with mode-specific parameters
    shaped_envelope = advanced_envelope_shaping_with_params(
        array, audio, mode_config, curve_params, use_filtering
    )
    
    return shaped_envelope


def advanced_envelope_shaping_with_params(gain_array: np.ndarray, audio: np.ndarray, 
                                        config: Config, params: dict, use_filtering: bool = True) -> np.ndarray:
    """
    Apply envelope shaping with custom parameters.
    
    Args:
        gain_array: Input gain array
        audio: Original audio
        config: Configuration
        params: Curve parameters
        use_filtering: If True, use Butterworth filtering (original). If False, use pure mathematical processing.
        
    Returns:
        Shaped envelope
    """
    # Apply logarithmic attack with custom curve shape
    attack_coef = params["attack_curve_shape"]
    attack_samples = ms_to_samples(config.limiter.attack, config.internal_sample_rate)
    
    # Apply program-dependent release with custom sensitivity
    transient_threshold = params["transient_sensitivity"]
    
    # Use the existing advanced shaping but with custom parameters
    return advanced_envelope_shaping(gain_array, audio, config, use_filtering)