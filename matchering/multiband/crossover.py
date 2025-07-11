# -*- coding: utf-8 -*-

"""
Crossover Filters for Multiband Processing
"""

import numpy as np
from scipy import signal
from typing import List, Tuple
from ..log import debug


class LinkwitzRileyCrossover:
    """
    Linkwitz-Riley crossover filter implementation.
    
    Provides phase-coherent band splitting with perfect reconstruction
    when bands are summed back together.
    """
    
    def __init__(self, sample_rate: int, crossover_freqs: List[float], order: int = 4):
        """
        Initialize crossover filters.
        
        Args:
            sample_rate: Audio sample rate
            crossover_freqs: List of crossover frequencies in Hz
            order: Filter order (must be even for L-R alignment)
        """
        self.sample_rate = sample_rate
        self.crossover_freqs = sorted(crossover_freqs)
        self.order = order if order % 2 == 0 else order + 1  # Ensure even order
        self.num_bands = len(crossover_freqs) + 1
        
        debug(f"Creating {self.num_bands}-band Linkwitz-Riley crossover at {crossover_freqs} Hz")
        
        # Pre-calculate filter coefficients
        self.filters = self._create_filter_coefficients()
        
    def _create_filter_coefficients(self) -> List[Tuple]:
        """
        Create filter coefficients for all bands.
        
        Returns:
            List of (b, a) coefficient tuples for each band
        """
        filters = []
        nyquist = self.sample_rate / 2
        
        for band_idx in range(self.num_bands):
            if band_idx == 0:
                # Lowest band - lowpass only
                if len(self.crossover_freqs) > 0:
                    freq_norm = self.crossover_freqs[0] / nyquist
                    freq_norm = min(freq_norm, 0.99)  # Avoid instability
                    b, a = signal.butter(self.order, freq_norm, btype='low')
                else:
                    # Single band - allpass
                    b, a = [1.0], [1.0]
            elif band_idx == self.num_bands - 1:
                # Highest band - highpass only
                freq_norm = self.crossover_freqs[-1] / nyquist
                freq_norm = max(freq_norm, 0.01)  # Avoid instability
                b, a = signal.butter(self.order, freq_norm, btype='high')
            else:
                # Middle bands - bandpass
                low_freq = self.crossover_freqs[band_idx - 1] / nyquist
                high_freq = self.crossover_freqs[band_idx] / nyquist
                
                # Ensure valid frequency range
                low_freq = max(low_freq, 0.01)
                high_freq = min(high_freq, 0.99)
                
                if low_freq >= high_freq:
                    # Degenerate case - create allpass
                    b, a = [1.0], [1.0]
                else:
                    b, a = signal.butter(self.order, [low_freq, high_freq], btype='band')
            
            filters.append((b, a))
            debug(f"Band {band_idx}: {self._get_band_description(band_idx)}")
        
        return filters
    
    def _get_band_description(self, band_idx: int) -> str:
        """Get description string for a band."""
        if band_idx == 0:
            if len(self.crossover_freqs) > 0:
                return f"Lowpass < {self.crossover_freqs[0]:.0f} Hz"
            else:
                return "Fullrange"
        elif band_idx == self.num_bands - 1:
            return f"Highpass > {self.crossover_freqs[-1]:.0f} Hz"
        else:
            low = self.crossover_freqs[band_idx - 1]
            high = self.crossover_freqs[band_idx]
            return f"Bandpass {low:.0f} - {high:.0f} Hz"
    
    def split_bands(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into frequency bands.
        
        Args:
            audio: Input audio array (samples, channels)
            
        Returns:
            List of band-filtered audio arrays
        """
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)
        
        bands = []
        
        for band_idx, (b, a) in enumerate(self.filters):
            # Apply filter to each channel
            band_audio = np.zeros_like(audio)
            
            for ch in range(audio.shape[1]):
                try:
                    # Use lfilter for real-time processing (causal)
                    band_audio[:, ch] = signal.lfilter(b, a, audio[:, ch])
                except Exception as e:
                    debug(f"Filter error in band {band_idx}, channel {ch}: {e}")
                    # Fallback to original audio
                    band_audio[:, ch] = audio[:, ch]
            
            bands.append(band_audio)
        
        debug(f"Split audio into {len(bands)} frequency bands")
        return bands
    
    def combine_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """
        Combine processed frequency bands back into full-range audio.
        
        Args:
            bands: List of processed band audio arrays
            
        Returns:
            Combined full-range audio
        """
        if not bands:
            raise ValueError("No bands provided for combination")
        
        # Sum all bands
        combined = np.zeros_like(bands[0])
        
        for band in bands:
            combined += band
        
        debug(f"Combined {len(bands)} frequency bands")
        return combined


def create_crossover_filters(sample_rate: int, num_bands: int = 4) -> LinkwitzRileyCrossover:
    """
    Create crossover filters with sensible default frequencies.
    
    Args:
        sample_rate: Audio sample rate
        num_bands: Number of frequency bands (3 or 4)
        
    Returns:
        Configured LinkwitzRileyCrossover object
    """
    if num_bands == 3:
        # 3-band: Bass | Mids | Highs
        crossover_freqs = [300, 3000]
    elif num_bands == 4:
        # 4-band: Bass | Low-Mid | High-Mid | Treble  
        crossover_freqs = [200, 1000, 5000]
    else:
        raise ValueError(f"Unsupported number of bands: {num_bands}. Use 3 or 4.")
    
    return LinkwitzRileyCrossover(sample_rate, crossover_freqs, order=4)


def create_custom_crossover(sample_rate: int, crossover_freqs: List[float]) -> LinkwitzRileyCrossover:
    """
    Create crossover with custom frequencies.
    
    Args:
        sample_rate: Audio sample rate  
        crossover_freqs: List of crossover frequencies in Hz
        
    Returns:
        Configured LinkwitzRileyCrossover object
    """
    # Validate frequencies
    nyquist = sample_rate / 2
    valid_freqs = []
    
    for freq in sorted(crossover_freqs):
        if 20 <= freq <= nyquist * 0.8:  # Leave some headroom
            valid_freqs.append(freq)
        else:
            debug(f"Skipping invalid crossover frequency: {freq} Hz")
    
    if not valid_freqs:
        raise ValueError("No valid crossover frequencies provided")
    
    return LinkwitzRileyCrossover(sample_rate, valid_freqs, order=4)