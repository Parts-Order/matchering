# -*- coding: utf-8 -*-

"""
Frequency Band Management for Multiband Compressor
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CompressorBand:
    """Configuration for a single compressor band."""
    
    name: str
    freq_low: float  # Hz
    freq_high: float  # Hz
    threshold: float  # dB
    ratio: float  # compression ratio (1.0 = no compression)
    attack: float  # ms
    release: float  # ms
    makeup_gain: float = 0.0  # dB
    knee: float = 2.0  # dB (soft knee width)
    enabled: bool = True
    
    def __post_init__(self):
        """Validate band parameters."""
        if self.freq_low >= self.freq_high:
            raise ValueError(f"Low frequency ({self.freq_low}) must be less than high frequency ({self.freq_high})")
        if self.ratio < 1.0:
            raise ValueError(f"Compression ratio ({self.ratio}) must be >= 1.0")
        if self.attack < 0.1 or self.attack > 1000:
            raise ValueError(f"Attack time ({self.attack}ms) must be between 0.1 and 1000ms")
        if self.release < 1.0 or self.release > 10000:
            raise ValueError(f"Release time ({self.release}ms) must be between 1.0 and 10000ms")


def create_default_bands() -> List[CompressorBand]:
    """
    Create default 4-band multiband compressor setup.
    
    Returns:
        List of CompressorBand objects with sensible defaults
    """
    return [
        CompressorBand(
            name="Bass",
            freq_low=20,
            freq_high=200,
            threshold=-12.0,
            ratio=3.0,
            attack=10.0,
            release=100.0,
            makeup_gain=2.0,
            knee=3.0
        ),
        CompressorBand(
            name="Low_Mid", 
            freq_low=200,
            freq_high=1000,
            threshold=-15.0,
            ratio=2.5,
            attack=5.0,
            release=80.0,
            makeup_gain=1.0,
            knee=2.5
        ),
        CompressorBand(
            name="High_Mid",
            freq_low=1000,
            freq_high=5000,
            threshold=-18.0,
            ratio=2.0,
            attack=3.0,
            release=60.0,
            makeup_gain=0.5,
            knee=2.0
        ),
        CompressorBand(
            name="Treble",
            freq_low=5000,
            freq_high=20000,
            threshold=-20.0,
            ratio=1.8,
            attack=1.0,
            release=40.0,
            makeup_gain=0.0,
            knee=1.5
        )
    ]


def create_3_band_setup() -> List[CompressorBand]:
    """
    Create 3-band setup for simpler processing.
    
    Returns:
        List of 3 CompressorBand objects
    """
    return [
        CompressorBand(
            name="Bass",
            freq_low=20,
            freq_high=300,
            threshold=-10.0,
            ratio=3.5,
            attack=15.0,
            release=120.0,
            makeup_gain=2.5,
            knee=4.0
        ),
        CompressorBand(
            name="Mids",
            freq_low=300,
            freq_high=3000,
            threshold=-15.0,
            ratio=2.8,
            attack=5.0,
            release=80.0,
            makeup_gain=1.5,
            knee=3.0
        ),
        CompressorBand(
            name="Highs",
            freq_low=3000,
            freq_high=20000,
            threshold=-18.0,
            ratio=2.2,
            attack=2.0,
            release=50.0,
            makeup_gain=1.0,
            knee=2.0
        )
    ]


def create_broadcast_preset() -> List[CompressorBand]:
    """
    Create broadcast-style multiband compression.
    
    Returns:
        List of CompressorBand objects optimized for broadcast
    """
    return [
        CompressorBand(
            name="Bass",
            freq_low=20,
            freq_high=150,
            threshold=-8.0,
            ratio=4.0,
            attack=20.0,
            release=200.0,
            makeup_gain=3.0,
            knee=4.0
        ),
        CompressorBand(
            name="Low_Mid",
            freq_low=150,
            freq_high=800,
            threshold=-12.0,
            ratio=3.0,
            attack=8.0,
            release=120.0,
            makeup_gain=2.0,
            knee=3.5
        ),
        CompressorBand(
            name="High_Mid",
            freq_low=800,
            freq_high=4000,
            threshold=-16.0,
            ratio=2.5,
            attack=4.0,
            release=80.0,
            makeup_gain=1.5,
            knee=2.5
        ),
        CompressorBand(
            name="Treble",
            freq_low=4000,
            freq_high=20000,
            threshold=-20.0,
            ratio=2.0,
            attack=1.5,
            release=30.0,
            makeup_gain=1.0,
            knee=2.0
        )
    ]


def create_mastering_preset() -> List[CompressorBand]:
    """
    Create gentle mastering-style multiband compression.
    
    Returns:
        List of CompressorBand objects optimized for mastering
    """
    return [
        CompressorBand(
            name="Bass",
            freq_low=20,
            freq_high=250,
            threshold=-15.0,
            ratio=2.5,
            attack=25.0,
            release=150.0,
            makeup_gain=1.5,
            knee=3.0
        ),
        CompressorBand(
            name="Low_Mid",
            freq_low=250,
            freq_high=1200,
            threshold=-18.0,
            ratio=2.0,
            attack=10.0,
            release=100.0,
            makeup_gain=1.0,
            knee=2.5
        ),
        CompressorBand(
            name="High_Mid", 
            freq_low=1200,
            freq_high=6000,
            threshold=-20.0,
            ratio=1.8,
            attack=5.0,
            release=70.0,
            makeup_gain=0.5,
            knee=2.0
        ),
        CompressorBand(
            name="Treble",
            freq_low=6000,
            freq_high=20000,
            threshold=-22.0,
            ratio=1.5,
            attack=2.0,
            release=50.0,
            makeup_gain=0.0,
            knee=1.5
        )
    ]


def get_preset_by_name(preset_name: str) -> List[CompressorBand]:
    """
    Get a preset by name.
    
    Args:
        preset_name: "default", "3band", "broadcast", or "mastering"
        
    Returns:
        List of CompressorBand objects
    """
    presets = {
        "default": create_default_bands,
        "3band": create_3_band_setup,
        "broadcast": create_broadcast_preset,
        "mastering": create_mastering_preset
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
    
    return presets[preset_name]()