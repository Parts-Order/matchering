# -*- coding: utf-8 -*-

"""
Multiband Compressor Implementation
"""

import numpy as np
from typing import List, Optional
from scipy import signal

from ..log import debug
from ..utils import ms_to_samples
from .crossover import LinkwitzRileyCrossover, create_crossover_filters
from .bands import CompressorBand, create_default_bands


class SingleBandCompressor:
    """
    Single-band compressor with soft knee and program-dependent release.
    """
    
    def __init__(self, band: CompressorBand, sample_rate: int):
        """
        Initialize single band compressor.
        
        Args:
            band: CompressorBand configuration
            sample_rate: Audio sample rate
        """
        self.band = band
        self.sample_rate = sample_rate
        self.envelope_state = 0.0  # For envelope follower
        
        # Convert timing to samples
        self.attack_coef = self._time_to_coef(band.attack)
        self.release_coef = self._time_to_coef(band.release)
        
        debug(f"Initialized compressor for {band.name} band: "
              f"threshold={band.threshold}dB, ratio={band.ratio}:1")
    
    def _time_to_coef(self, time_ms: float) -> float:
        """Convert time constant to filter coefficient."""
        time_samples = ms_to_samples(time_ms, self.sample_rate)
        return np.exp(-1.0 / time_samples)
    
    def _soft_knee_compression(self, level_db: float) -> float:
        """
        Apply soft knee compression curve.
        
        Args:
            level_db: Input level in dB
            
        Returns:
            Gain reduction in dB
        """
        threshold = self.band.threshold
        ratio = self.band.ratio
        knee = self.band.knee
        
        # Soft knee implementation
        if level_db <= (threshold - knee/2):
            # Below knee - no compression
            return 0.0
        elif level_db >= (threshold + knee/2):
            # Above knee - full compression
            overshoot = level_db - threshold
            gain_reduction = overshoot * (1.0 - 1.0/ratio)
            return -gain_reduction
        else:
            # In knee region - smooth transition
            knee_ratio = (level_db - threshold + knee/2) / knee
            # Smooth S-curve using cubic interpolation
            smooth_ratio = knee_ratio * knee_ratio * (3.0 - 2.0 * knee_ratio)
            overshoot = level_db - threshold
            gain_reduction = overshoot * smooth_ratio * (1.0 - 1.0/ratio)
            return -gain_reduction
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through single band compressor.
        
        Args:
            audio: Input audio array
            
        Returns:
            Compressed audio
        """
        if not self.band.enabled:
            return audio
        
        # Convert to dB for level detection
        audio_abs = np.abs(audio)
        audio_abs = np.maximum(audio_abs, 1e-10)  # Avoid log(0)
        level_db = 20 * np.log10(audio_abs)
        
        # Apply compression to each sample
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get current level
            current_level = level_db[i] if len(level_db.shape) == 1 else np.max(level_db[i])
            
            # Calculate target gain reduction
            target_gain_reduction = self._soft_knee_compression(current_level)
            
            # Envelope follower with attack/release
            if target_gain_reduction < self.envelope_state:
                # Attack (faster response to gain reduction)
                self.envelope_state = (self.attack_coef * self.envelope_state + 
                                     (1.0 - self.attack_coef) * target_gain_reduction)
            else:
                # Release (slower response to gain increase)
                self.envelope_state = (self.release_coef * self.envelope_state + 
                                     (1.0 - self.release_coef) * target_gain_reduction)
            
            # Apply gain reduction and makeup gain
            total_gain_db = self.envelope_state + self.band.makeup_gain
            gain_linear = 10**(total_gain_db / 20)
            
            # Apply gain to all channels
            if len(audio.shape) == 1:
                output[i] = audio[i] * gain_linear
            else:
                output[i] = audio[i] * gain_linear
        
        return output


class MultibandCompressor:
    """
    Multiband compressor with Linkwitz-Riley crossovers.
    """
    
    def __init__(self, bands: List[CompressorBand], sample_rate: int):
        """
        Initialize multiband compressor.
        
        Args:
            bands: List of CompressorBand configurations
            sample_rate: Audio sample rate
        """
        self.bands = bands
        self.sample_rate = sample_rate
        
        # Create crossover frequencies from band definitions
        crossover_freqs = []
        for i in range(len(bands) - 1):
            # Use geometric mean of adjacent band boundaries
            freq = np.sqrt(bands[i].freq_high * bands[i + 1].freq_low)
            crossover_freqs.append(freq)
        
        debug(f"Multiband compressor: {len(bands)} bands, crossovers at {crossover_freqs} Hz")
        
        # Create crossover filters
        self.crossover = LinkwitzRileyCrossover(sample_rate, crossover_freqs, order=4)
        
        # Create individual band compressors
        self.compressors = [SingleBandCompressor(band, sample_rate) for band in bands]
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through multiband compressor.
        
        Args:
            audio: Input audio array
            
        Returns:
            Multiband compressed audio
        """
        debug("Processing audio through multiband compressor")
        
        # Split into frequency bands
        band_signals = self.crossover.split_bands(audio)
        
        # Process each band
        processed_bands = []
        for i, (band_signal, compressor) in enumerate(zip(band_signals, self.compressors)):
            processed_band = compressor.process(band_signal)
            processed_bands.append(processed_band)
            debug(f"Processed band {i} ({compressor.band.name})")
        
        # Combine bands
        output = self.crossover.combine_bands(processed_bands)
        
        debug("Multiband compression completed")
        return output
    
    def get_band_info(self) -> List[dict]:
        """
        Get information about each band.
        
        Returns:
            List of dictionaries with band information
        """
        info = []
        for i, band in enumerate(self.bands):
            info.append({
                "index": i,
                "name": band.name,
                "frequency_range": f"{band.freq_low:.0f} - {band.freq_high:.0f} Hz",
                "threshold": f"{band.threshold:.1f} dB",
                "ratio": f"{band.ratio:.1f}:1",
                "attack": f"{band.attack:.1f} ms",
                "release": f"{band.release:.1f} ms",
                "makeup_gain": f"{band.makeup_gain:.1f} dB",
                "enabled": band.enabled
            })
        return info


def compress_multiband(audio: np.ndarray, sample_rate: int, 
                      preset: str = "default", 
                      bands: Optional[List[CompressorBand]] = None) -> np.ndarray:
    """
    Apply multiband compression to audio.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        preset: Preset name ("default", "3band", "broadcast", "mastering")
        bands: Custom band configurations (overrides preset)
        
    Returns:
        Multiband compressed audio
    """
    from .bands import get_preset_by_name
    
    # Use custom bands or load preset
    if bands is not None:
        compressor_bands = bands
    else:
        compressor_bands = get_preset_by_name(preset)
    
    # Create and run compressor
    compressor = MultibandCompressor(compressor_bands, sample_rate)
    
    debug(f"Applying multiband compression with {len(compressor_bands)} bands")
    return compressor.process(audio)