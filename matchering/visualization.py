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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from .log import debug

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_waveform_plot(audio: np.ndarray, sample_rate: int, output_path: str = None) -> str:
    """
    Create waveform visualization.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping waveform plot")
        return None
    
    # Professional figure setup
    plt.figure(figsize=(14, 8))
    
    # Downsample audio for faster plotting
    downsample_factor = max(1, len(audio) // 10000)
    audio_ds = audio[::downsample_factor]
    
    # Time axis
    time = np.arange(len(audio_ds)) / sample_rate * downsample_factor
    
    # Plot waveform
    if len(audio.shape) == 1:
        plt.plot(time, audio_ds, linewidth=0.5, color='blue', alpha=0.8)
        plt.ylabel('Amplitude')
    else:
        # Stereo
        plt.plot(time, audio_ds[:, 0], linewidth=0.5, color='blue', alpha=0.8, label='Left')
        plt.plot(time, audio_ds[:, 1], linewidth=0.5, color='red', alpha=0.8, label='Right')
        plt.legend(loc='upper right', fontsize=10)
        plt.ylabel('Amplitude')
    
    plt.xlabel('Time (seconds)')
    plt.title('Waveform Analysis', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    
    if output_path is None:
        output_path = "waveform.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    debug(f"Waveform plot saved to: {output_path}")
    return output_path


def create_spectrum_plot(freqs: np.ndarray, spectrum: np.ndarray, output_path: str = None) -> str:
    """
    Create frequency spectrum visualization.
    
    Args:
        freqs: Frequency array
        spectrum: Spectrum array (in dB)
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping spectrum plot")
        return None
    
    # Professional figure setup
    plt.figure(figsize=(14, 8))
    
    # Plot spectrum
    plt.semilogx(freqs[1:], spectrum[1:], linewidth=1.5, color='blue', alpha=0.8)
    
    # Add frequency band markers
    band_freqs = [60, 250, 500, 2000, 4000, 6000]
    band_names = ['Bass', 'Low Mid', 'Mid', 'High Mid', 'Presence', 'Brilliance']
    
    for freq, name in zip(band_freqs, band_names):
        if freq <= freqs[-1]:
            plt.axvline(x=freq, color='red', linestyle='--', alpha=0.6, linewidth=1)
            plt.text(freq, plt.ylim()[1] - 5, name, rotation=90, 
                   verticalalignment='top', fontsize=10, fontweight='bold')
    
    plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    plt.ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
    plt.title('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(20, freqs[-1])
    plt.tight_layout()
    
    if output_path is None:
        output_path = "spectrum.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    debug(f"Spectrum plot saved to: {output_path}")
    return output_path


def create_lufs_meter_plot(lufs: float, lra: float, output_path: str = None) -> str:
    """
    Create LUFS meter visualization.
    
    Args:
        lufs: LUFS value
        lra: Loudness Range value
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping LUFS meter plot")
        return None
    
    # Professional figure setup with BLACK text on WHITE background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor='white')
    
    # LUFS meter
    ax1.barh(0, 60, height=0.5, color='lightgray', alpha=0.3)
    
    # Add LUFS zones with darker colors for visibility
    zones = [
        (-60, -23, '#cc0000', 'Too Quiet'),
        (-23, -18, '#ffcc00', 'Broadcast'),
        (-18, -14, '#00cc00', 'Streaming'),
        (-14, 0, '#cc0000', 'Too Loud')
    ]
    
    for start, end, color, label in zones:
        width = end - start
        ax1.barh(0, width, left=start + 60, height=0.5, 
                color=color, alpha=0.7, label=label)
    
    # Add LUFS indicator in BLACK
    if lufs > -np.inf:
        ax1.axvline(x=lufs + 60, color='black', linewidth=4, label=f'LUFS: {lufs:.1f}')
    
    ax1.set_xlim(0, 60)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('LUFS', fontsize=12, fontweight='bold', color='black')
    ax1.set_title('Loudness Meter', fontsize=14, fontweight='bold', color='black')
    ax1.set_yticks([])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.tick_params(colors='black')
    
    # Set x-axis labels in BLACK
    ax1.set_xticks([0, 15, 30, 45, 60])
    ax1.set_xticklabels(['-60', '-45', '-30', '-15', '0'], color='black', fontweight='bold')
    
    # LRA meter
    lra_max = 30
    ax2.barh(0, lra_max, height=0.5, color='lightgray', alpha=0.3)
    
    # Add LRA zones with darker colors
    lra_zones = [
        (0, 7, '#cc0000', 'Over-compressed'),
        (7, 15, '#ffcc00', 'Moderate'),
        (15, 20, '#00cc00', 'Good'),
        (20, 30, '#cc0000', 'Too Dynamic')
    ]
    
    for start, end, color, label in lra_zones:
        width = end - start
        ax2.barh(0, width, left=start, height=0.5, 
                color=color, alpha=0.7, label=label)
    
    # Add LRA indicator in BLACK
    if lra > 0:
        ax2.axvline(x=min(lra, lra_max), color='black', linewidth=4, 
                   label=f'LRA: {lra:.1f} LU')
    
    ax2.set_xlim(0, lra_max)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('LRA (LU)', fontsize=12, fontweight='bold', color='black')
    ax2.set_title('Loudness Range', fontsize=14, fontweight='bold', color='black')
    ax2.set_yticks([])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.tick_params(colors='black')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = "lufs_meter.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    debug(f"LUFS meter plot saved to: {output_path}")
    return output_path


def create_dynamics_plot(audio: np.ndarray, sample_rate: int, output_path: str = None) -> str:
    """
    Create dynamics visualization showing envelope and transients.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping dynamics plot")
        return None
    
    from .analysis import _calculate_envelope, _detect_transients
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time axis
    time = np.arange(len(audio)) / sample_rate
    
    # Top plot: Waveform and envelope
    if len(audio.shape) == 1:
        audio_mono = audio
    else:
        audio_mono = np.mean(audio, axis=1)
    
    envelope = _calculate_envelope(audio_mono, sample_rate)
    
    ax1.plot(time, audio_mono, linewidth=0.3, color='blue', alpha=0.5, label='Waveform')
    ax1.plot(time, envelope, linewidth=1, color='red', label='Envelope')
    ax1.plot(time, -envelope, linewidth=1, color='red')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform and Envelope')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: RMS vs Peak Analysis (more useful than transient lines)
    # Calculate windowed RMS and peak levels
    window_size = max(1024, int(0.1 * sample_rate))  # 100ms windows
    hop_size = window_size // 4
    
    rms_levels = []
    peak_levels = []
    time_windows = []
    
    for i in range(0, len(audio_mono) - window_size, hop_size):
        window = audio_mono[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        peak = np.abs(window).max()
        
        rms_levels.append(20 * np.log10(rms) if rms > 0 else -60)
        peak_levels.append(20 * np.log10(peak) if peak > 0 else -60)
        time_windows.append((i + window_size/2) / sample_rate)
    
    # Plot RMS and Peak levels
    ax2.plot(time_windows, rms_levels, linewidth=1.5, color='blue', label='RMS Level', alpha=0.8)
    ax2.plot(time_windows, peak_levels, linewidth=1, color='red', label='Peak Level', alpha=0.8)
    
    # Fill between for crest factor visualization
    ax2.fill_between(time_windows, rms_levels, peak_levels, alpha=0.2, color='orange', label='Crest Factor')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Level (dB)')
    ax2.set_title('Dynamic Range Analysis (RMS vs Peak)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, 0)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = "dynamics.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    debug(f"Dynamics plot saved to: {output_path}")
    return output_path


def create_stereo_plot(audio: np.ndarray, sample_rate: int = 44100, output_path: str = None) -> str:
    """
    Create comprehensive stereo field visualization.
    
    Args:
        audio: Input audio array (must be stereo)
        sample_rate: Sample rate (default: 44100)
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping stereo plot")
        return None
    
    if len(audio.shape) != 2 or audio.shape[1] != 2:
        debug("Audio must be stereo for stereo plot")
        return None
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    left = audio[:, 0]
    right = audio[:, 1]
    
    # Downsample aggressively for visualization
    downsample_factor = max(1, len(left) // 2000)
    left_ds = left[::downsample_factor]
    right_ds = right[::downsample_factor]
    
    # 1. Vectorscope/Goniometer (top-left) - FIXED TO ACTUALLY SHOW CONTENT
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create density plot instead of useless scatter
    # Bin the data to create a heatmap
    bins = 50
    H, xedges, yedges = np.histogram2d(left_ds, right_ds, bins=bins, range=[[-1, 1], [-1, 1]])
    
    # Only show bins with content
    H = np.ma.masked_where(H == 0, H)
    
    # Create heatmap
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im = ax1.pcolormesh(X, Y, H.T, cmap='Blues', alpha=0.8)
    
    # Add reference lines
    ax1.axhline(y=0, color='white', linestyle='-', alpha=0.8, linewidth=1)
    ax1.axvline(x=0, color='white', linestyle='-', alpha=0.8, linewidth=1)
    ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.8, linewidth=2, label='L=R (Mono)')
    ax1.plot([-1, 1], [1, -1], 'g--', alpha=0.8, linewidth=2, label='L=-R (Anti-phase)')
    
    # Add correlation info
    correlation = np.corrcoef(left, right)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))
    
    ax1.set_xlabel('Left Channel')
    ax1.set_ylabel('Right Channel')
    ax1.set_title('Vectorscope (Content Density)')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Mid/Side Energy Analysis (top-right) - FIXED TO SHOW ACTUAL CONTENT
    ax2 = fig.add_subplot(gs[0, 1])
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Calculate windowed M/S energy over time instead of useless scatter
    window_size = max(1024, int(0.1 * sample_rate))  # 100ms windows
    hop_size = window_size // 4
    
    mid_energy = []
    side_energy = []
    time_ms = []
    
    for i in range(0, len(mid) - window_size, hop_size):
        mid_window = mid[i:i + window_size]
        side_window = side[i:i + window_size]
        
        mid_rms = np.sqrt(np.mean(mid_window**2))
        side_rms = np.sqrt(np.mean(side_window**2))
        
        mid_energy.append(20 * np.log10(mid_rms) if mid_rms > 0 else -60)
        side_energy.append(20 * np.log10(side_rms) if side_rms > 0 else -60)
        time_ms.append((i + window_size/2) / sample_rate)
    
    # Plot M/S energy over time
    ax2.plot(time_ms, mid_energy, linewidth=1.5, color='blue', label='Mid Energy', alpha=0.8)
    ax2.plot(time_ms, side_energy, linewidth=1.5, color='red', label='Side Energy', alpha=0.8)
    
    # Fill between to show stereo width
    ax2.fill_between(time_ms, mid_energy, side_energy, alpha=0.2, color='green', label='Stereo Content')
    
    # Add overall stereo width info
    mid_rms_total = np.sqrt(np.mean(mid**2))
    side_rms_total = np.sqrt(np.mean(side**2))
    
    if mid_rms_total > 0 and side_rms_total > 0:
        stereo_width = 20 * np.log10(side_rms_total / mid_rms_total)
        ax2.text(0.05, 0.95, f'Avg Width: {stereo_width:.1f} dB', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Energy (dB)')
    ax2.set_title('Mid/Side Energy Over Time')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, 0)
    
    # 3. Phase analysis (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate instantaneous phase difference
    angles = np.arctan2(right_ds, left_ds) * 180 / np.pi
    
    # Histogram of phase angles
    ax3.hist(angles, bins=30, alpha=0.7, color='purple', edgecolor='none')
    ax3.set_xlabel('Phase Angle (degrees)')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Phase Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-180, 180)
    
    # Add statistics
    phase_std = np.std(angles)
    ax3.text(0.05, 0.95, f'Phase Spread: {phase_std:.1f}°', 
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Stereo width by frequency (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Simple frequency analysis for stereo width
    from .analysis import _analyze_frequency_dependent_stereo_width
    
    width_by_freq = _analyze_frequency_dependent_stereo_width(left, right, sample_rate)
    
    bands = list(width_by_freq.keys())
    widths = [width_by_freq[band] if width_by_freq[band] != -np.inf else -60 for band in bands]
    
    bars = ax4.bar(range(len(bands)), widths, color='orange', alpha=0.7)
    ax4.set_xticks(range(len(bands)))
    ax4.set_xticklabels([band.replace('_', ' ').title() for band in bands], rotation=45, ha='right')
    ax4.set_ylabel('Stereo Width (dB)')
    ax4.set_title('Stereo Width by Frequency')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Color code bars
    for bar, width in zip(bars, widths):
        if width > 6:
            bar.set_color('red')  # Too wide
        elif width > 0:
            bar.set_color('green')  # Good width
        elif width > -6:
            bar.set_color('yellow')  # Narrow
        else:
            bar.set_color('gray')  # Very narrow/mono
    
    # Overall title
    fig.suptitle('Comprehensive Stereo Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = "stereo.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    debug(f"Stereo plot saved to: {output_path}")
    return output_path


def create_analysis_summary_plot(analysis_data: dict, output_path: str = None) -> str:
    """
    Create comprehensive analysis summary visualization.
    
    Args:
        analysis_data: Analysis results from comprehensive_analysis()
        output_path: Output file path (if None, auto-generated)
        
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping analysis summary plot")
        return None
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Basic metrics (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    basic = analysis_data['basic_metrics']
    loudness = analysis_data['loudness_metrics']
    
    metrics = ['Peak', 'RMS', 'Crest Factor', 'LUFS', 'True Peak']
    values = [basic['peak_db'], basic['rms_db'], basic['crest_factor'], 
             loudness['lufs'], loudness['true_peak_db']]
    
    bars = ax1.barh(metrics, values)
    ax1.set_xlabel('dB')
    ax1.set_title('Basic Metrics')
    ax1.grid(True, alpha=0.3)
    
    # Color code bars
    colors = ['red' if v > -0.1 else 'orange' if v > -6 else 'green' for v in values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Frequency balance (top-center) - FIXED
    ax2 = fig.add_subplot(gs[0, 1])
    if 'frequency_metrics' in analysis_data and analysis_data['frequency_metrics']:
        freq_data = analysis_data['frequency_metrics']
        
        # Use per_channel data if available (stereo), otherwise band_energies
        if 'per_channel' in freq_data and freq_data['per_channel']:
            # Stereo - show L/R frequency balance
            left_bands = freq_data['per_channel']['left']['band_energies']
            right_bands = freq_data['per_channel']['right']['band_energies']
            
            band_names = list(left_bands.keys())
            left_values = [left_bands[name] for name in band_names]
            right_values = [right_bands[name] for name in band_names]
            
            x = np.arange(len(band_names))
            width = 0.35
            
            ax2.bar(x - width/2, left_values, width, label='Left', color='blue', alpha=0.7)
            ax2.bar(x + width/2, right_values, width, label='Right', color='red', alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels([name.replace('_', ' ').title() for name in band_names], 
                               rotation=45, ha='right')
            ax2.legend()
            ax2.set_title('L/R Frequency Balance')
        elif 'band_energies' in freq_data:
            # Mono - show single frequency balance
            bands = freq_data['band_energies']
            band_names = list(bands.keys())
            band_values = [bands[name] for name in band_names]
            
            bars = ax2.bar(range(len(band_names)), band_values, color='blue', alpha=0.7)
            ax2.set_xticks(range(len(band_names)))
            ax2.set_xticklabels([name.replace('_', ' ').title() for name in band_names], 
                               rotation=45, ha='right')
            ax2.set_title('Frequency Balance')
            
            # Color code bars relative to average
            avg_energy = np.mean(band_values)
            for bar, value in zip(bars, band_values):
                if value > avg_energy + 3:
                    bar.set_color('red')
                elif value < avg_energy - 3:
                    bar.set_color('orange')
        else:
            ax2.text(0.5, 0.5, 'No frequency data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Frequency Balance')
    else:
        ax2.text(0.5, 0.5, 'No frequency data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Frequency Balance')
    
    ax2.set_ylabel('Energy (dB)')
    ax2.grid(True, alpha=0.3)
    
    # Dynamics (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    dynamics = analysis_data['dynamic_metrics']
    
    dynamic_metrics = ['DR14', 'LRA', 'Punch Score']
    dynamic_values = [dynamics['dr14'], loudness['lra'], 
                     dynamics['punch_metrics']['punch_score']]
    
    ax3.bar(dynamic_metrics, dynamic_values)
    ax3.set_ylabel('Value')
    ax3.set_title('Dynamic Range')
    ax3.grid(True, alpha=0.3)
    
    # LUFS Meter (middle-left) - REPLACED USELESS COMPLIANCE CHART
    ax4 = fig.add_subplot(gs[1, 0])
    if 'loudness_metrics' in analysis_data:
        loudness = analysis_data['loudness_metrics']
        lufs = loudness.get('lufs', -np.inf)
        lra = loudness.get('lra', 0)
        
        # Create LUFS scale
        lufs_scale = np.linspace(-30, -10, 100)
        lufs_positions = np.linspace(0, 1, 100)
        
        # Color zones
        for i, level in enumerate(lufs_scale):
            if level < -23:
                color = 'red'
            elif level < -18:
                color = 'yellow' 
            elif level < -14:
                color = 'green'
            else:
                color = 'orange'
            
            ax4.barh(0, 0.01, left=lufs_positions[i], height=0.3, color=color, alpha=0.7)
        
        # Add LUFS indicator
        if lufs > -np.inf:
            lufs_pos = (lufs + 30) / 20  # Normalize to 0-1
            lufs_pos = np.clip(lufs_pos, 0, 1)
            ax4.axvline(x=lufs_pos, color='black', linewidth=3)
            ax4.text(lufs_pos, 0.5, f'{lufs:.1f}\nLUFS', ha='center', va='center', 
                    fontweight='bold', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_title('LUFS Meter')
        ax4.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax4.set_xticklabels(['-30', '-25', '-20', '-15', '-10'])
        ax4.set_xlabel('LUFS')
        ax4.set_yticks([])
    else:
        ax4.text(0.5, 0.5, 'No loudness data', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('LUFS Meter')
    
    # Spectrum plot (middle-center and middle-right)
    ax5 = fig.add_subplot(gs[1, 1:])
    if 'freqs' in analysis_data['frequency_metrics']:
        freqs = analysis_data['frequency_metrics']['freqs']
        spectrum = analysis_data['frequency_metrics']['avg_spectrum']
        
        ax5.semilogx(freqs[1:], spectrum[1:], linewidth=1, color='blue')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Magnitude (dB)')
        ax5.set_title('Frequency Spectrum')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(20, freqs[-1])
    
    # Enhanced stereo information (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    if analysis_data['stereo_metrics'] and 'error' not in analysis_data['stereo_metrics']:
        stereo = analysis_data['stereo_metrics']
        
        # Basic stereo info
        stereo_info = [
            f"L/R Balance: {stereo.get('lr_balance', 0):.1f} dB",
            f"Stereo Width: {stereo.get('stereo_width', 0):.1f} dB",
            f"Correlation: {stereo.get('correlation', 0):.3f}",
            f"Phase Coherence: {stereo.get('phase_coherence', 0):.3f}",
            f"Mono Compatibility: {stereo.get('mono_rms', 0):.1f} dB"
        ]
        
        # Add M/S analysis if available
        if 'ms_analysis' in stereo:
            ms_data = stereo['ms_analysis']
            stereo_info.extend([
                f"M/S Ratio: {ms_data.get('ms_ratio', 0):.2f}",
                f"Mid Energy: {ms_data.get('mid_energy_percent', 0):.1f}%",
                f"Side Energy: {ms_data.get('side_energy_percent', 0):.1f}%",
                f"Stereo Classification: {ms_data.get('stereo_content_classification', 'Unknown')}"
            ])
        
        # Add phase analysis if available
        if 'phase_analysis' in stereo:
            phase_data = stereo['phase_analysis']
            stereo_info.extend([
                f"Phase Delay: {phase_data.get('phase_delay_ms', 0):.2f} ms",
                f"Polarity: {phase_data.get('polarity_status', 'Unknown')}"
            ])
        
        # Create two-column layout
        col1_info = stereo_info[:len(stereo_info)//2]
        col2_info = stereo_info[len(stereo_info)//2:]
        
        ax6.text(0.05, 0.5, '\n'.join(col1_info), transform=ax6.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax6.text(0.55, 0.5, '\n'.join(col2_info), transform=ax6.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        ax6.set_title('Comprehensive Stereo Analysis')
        ax6.axis('off')
    else:
        ax6.text(0.5, 0.5, 'Mono Audio - No Stereo Analysis', 
                transform=ax6.transAxes, fontsize=14, 
                horizontalalignment='center', verticalalignment='center')
        ax6.set_title('Stereo Characteristics')
        ax6.axis('off')
    
    # Overall title
    fig.suptitle('Comprehensive Audio Analysis', fontsize=16, fontweight='bold')
    
    if output_path is None:
        output_path = "analysis_summary.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    debug(f"Analysis summary plot saved to: {output_path}")
    return output_path


def create_all_plots(audio: np.ndarray, sample_rate: int, analysis_data: dict = None, 
                    output_dir: str = "analysis_plots") -> dict:
    """
    Create all available plots for audio analysis with parallel processing.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        analysis_data: Analysis results (if None, will be computed)
        output_dir: Output directory for plots
        
    Returns:
        Dictionary with paths to all created plots
    """
    if not MATPLOTLIB_AVAILABLE:
        debug("Matplotlib not available, skipping all plots")
        return {}
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    debug(f"Creating analysis plots in: {output_path}")
    
    # Compute analysis if not provided
    if analysis_data is None:
        from .analysis import comprehensive_analysis
        analysis_data = comprehensive_analysis(audio, sample_rate)
    
    # Create plotting tasks for parallel execution
    plot_tasks = [
        ("waveform", partial(create_waveform_plot, audio, sample_rate, str(output_path / "waveform.png"))),
        ("dynamics", partial(create_dynamics_plot, audio, sample_rate, str(output_path / "dynamics.png"))),
        ("summary", partial(create_analysis_summary_plot, analysis_data, str(output_path / "summary.png")))
    ]
    
    # Add spectrum plot if data available
    if 'frequency_metrics' in analysis_data:
        freq_data = analysis_data['frequency_metrics']
        if 'freqs' in freq_data and 'avg_spectrum' in freq_data:
            plot_tasks.append(("spectrum", partial(create_spectrum_plot, freq_data['freqs'], 
                                                 freq_data['avg_spectrum'], str(output_path / "spectrum.png"))))
    
    # Add LUFS meter if data available
    if 'loudness_metrics' in analysis_data:
        loudness = analysis_data['loudness_metrics']
        plot_tasks.append(("lufs_meter", partial(create_lufs_meter_plot, loudness['lufs'], 
                                                loudness['lra'], str(output_path / "lufs_meter.png"))))
    
    # Add stereo plot if stereo audio
    if len(audio.shape) == 2 and audio.shape[1] == 2:
        plot_tasks.append(("stereo", partial(create_stereo_plot, audio, sample_rate, str(output_path / "stereo.png"))))
    
    # Execute plotting tasks sequentially to avoid matplotlib issues
    plots = {}
    for name, task in plot_tasks:
        try:
            result = task()
            if result is not None:
                plots[name] = result
                debug(f"Created {name} plot: {result}")
        except Exception as e:
            debug(f"Error creating {name} plot: {e}")
    
    debug(f"Created {len(plots)} analysis plots")
    return plots