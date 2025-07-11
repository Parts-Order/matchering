#!/usr/bin/env python3
"""
Hyrax Master Bus Limiter

Sums stems from a directory and applies Hyrax limiter to the master bus,
extracting both the limited master and the sidechain signal.

Usage: python hyrax_sidechain_extract.py <stems_directory>
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import glob

# Import matchering components
import matchering as mg
from matchering.limiter.hyrax import limit
from matchering.limiter.enhanced import limit_with_oversampling, get_limiter_analysis
from matchering.lufs import measure_lufs, check_atmos_compliance, normalize_to_lufs
from matchering.analysis import comprehensive_analysis
from matchering.visualization import create_all_plots
from matchering.loader import load
from matchering.defaults import Config
from matchering.dsp import normalize


def limit_with_sidechain(array: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified version of hyrax.limit() that returns both limited audio and gain envelope.
    """
    import math
    from scipy import signal
    from scipy.ndimage import maximum_filter1d
    from matchering.dsp import rectify, flip, max_mix
    from matchering.utils import make_odd, ms_to_samples
    
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
    
    # Use the same rectification as original
    rectified = rectify(array, config.threshold)
    
    if np.all(np.isclose(rectified, 1.0)):
        # No limiting needed - return original audio and unity gain envelope
        gain_envelope = np.ones(array.shape[0])
        return array, gain_envelope
    
    # Calculate gain envelope (same as original hyrax implementation)
    gain_hard_clip = flip(1.0 / rectified)
    gain_attack, gain_hard_clip_slided = __process_attack(np.copy(gain_hard_clip), config)
    gain_release = __process_release(np.copy(gain_hard_clip_slided), config)
    gain = flip(max_mix(gain_hard_clip, gain_attack, gain_release))
    
    # Apply limiting
    limited_audio = array * gain[:, None]
    
    # Convert gain envelope to audio signal (invert to hear gain reduction)
    sidechain_signal = 1.0 - gain
    
    return limited_audio, sidechain_signal


def _load_stem_file(stem_file: str) -> tuple[np.ndarray, int, str]:
    """Load a single stem file - for parallel processing."""
    try:
        audio, sr = sf.read(stem_file)
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)
        return audio, sr, stem_file
    except Exception as e:
        print(f"Error loading {stem_file}: {e}")
        return None, None, stem_file


def sum_stems(stems_dir: Path) -> tuple[np.ndarray, int]:
    """
    Sum all audio files in the stems directory to create a master mix using parallel loading.
    """
    # Find all audio files
    audio_extensions = ['*.wav', '*.aiff', '*.flac', '*.mp3']
    stem_files = []
    for ext in audio_extensions:
        stem_files.extend(glob.glob(str(stems_dir / ext)))
    
    if not stem_files:
        raise ValueError(f"No audio files found in {stems_dir}")
    
    print(f"Found {len(stem_files)} stems:")
    for stem in stem_files:
        print(f"  - {Path(stem).name}")
    
    # Load all stems in parallel
    print("Loading stems in parallel...")
    from concurrent.futures import ThreadPoolExecutor
    
    loaded_stems = []
    sample_rate = None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Load all stems in parallel
        future_to_file = {executor.submit(_load_stem_file, stem_file): stem_file for stem_file in stem_files}
        
        for future in future_to_file:
            stem_file = future_to_file[future]
            try:
                audio, sr, filename = future.result()
                if audio is not None:
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        print(f"Warning: Sample rate mismatch in {filename} ({sr} vs {sample_rate})")
                        continue
                    
                    loaded_stems.append((audio, filename))
                    print(f"Loaded: {Path(filename).name}")
            except Exception as e:
                print(f"Error processing {stem_file}: {e}")
    
    if not loaded_stems:
        raise ValueError("No stems could be loaded successfully")
    
    # Find maximum length and channel count
    max_length = max(audio.shape[0] for audio, _ in loaded_stems)
    max_channels = max(audio.shape[1] for audio, _ in loaded_stems)
    
    print(f"Summing {len(loaded_stems)} stems...")
    
    # Initialize master mix
    master_mix = np.zeros((max_length, max_channels))
    
    # Sum all stems
    for audio, filename in loaded_stems:
        # Pad length if needed
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = np.pad(audio, ((0, padding), (0, 0)), mode='constant')
        
        # Expand channels if needed
        if audio.shape[1] < max_channels:
            if audio.shape[1] == 1 and max_channels == 2:
                audio = np.repeat(audio, 2, axis=1)
            else:
                # Pad with zeros for other channel configurations
                padding = max_channels - audio.shape[1]
                audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
        
        # Sum the stem
        master_mix += audio
    
    return master_mix, sample_rate


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyrax Master Bus Limiter with LUFS compliance")
    parser.add_argument("stems_directory", help="Directory containing stem files")
    parser.add_argument("--force-18-lufs", action="store_true", 
                       help="Force output to -18 LUFS for Atmos compliance")
    parser.add_argument("--analyze", action="store_true",
                       help="Perform comprehensive audio analysis and generate plots")
    parser.add_argument("--analysis-dir", default="analysis_plots",
                       help="Directory for analysis plots (default: analysis_plots)")
    parser.add_argument("--lookahead", type=float, default=5.0,
                       help="Lookahead buffer in milliseconds (0-20ms, default: 5.0)")
    parser.add_argument("--mode", choices=["transparent", "punchy", "aggressive"], default="transparent",
                       help="Limiting mode: transparent (smooth), punchy (fast transients), aggressive (tight control)")
    parser.add_argument("--multiband", action="store_true",
                       help="Enable multiband compression before limiting")
    parser.add_argument("--multiband-preset", choices=["default", "3band", "broadcast", "mastering"], 
                       default="default", help="Multiband compressor preset (default: default)")
    
    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    
    stems_dir = Path(args.stems_directory)
    if not stems_dir.exists() or not stems_dir.is_dir():
        print(f"Error: Directory '{stems_dir}' not found")
        sys.exit(1)
    
    # Sum all stems to create master mix
    print("Summing stems...")
    try:
        master_mix, sr = sum_stems(stems_dir)
    except Exception as e:
        print(f"Error summing stems: {e}")
        sys.exit(1)
    
    print(f"Master mix created: {master_mix.shape[0]} samples, {master_mix.shape[1]} channels, {sr}Hz")
    
    # Create config with default limiter settings
    config = Config()
    config.internal_sample_rate = sr
    
    # Normalize audio to prepare for limiting
    normalized_audio, _ = normalize(master_mix, config.threshold, config.min_value, False)
    
    # Analyze audio before processing
    print("Analyzing master mix...")
    analysis = get_limiter_analysis(normalized_audio, config)
    print(f"Peak: {analysis['peak_db']:.2f}dB, True Peak: {analysis['true_peak_db']:.2f}dB")
    print(f"Crest Factor: {analysis['crest_factor']:.1f}dB")
    print(f"Needs oversampling: {analysis['needs_oversampling']}")
    print(f"EBU R128 compliant: {analysis['ebu_compliant']}")
    
    # LUFS analysis
    print("Analyzing LUFS...")
    atmos_compliance = check_atmos_compliance(normalized_audio, sr)
    print(f"LUFS: {atmos_compliance['lufs']:.1f} LUFS")
    print(f"LRA: {atmos_compliance['lra']:.1f} LU")
    print(f"Atmos compliant: {atmos_compliance['atmos_compliant']}")
    print(f"Headroom to -18 LUFS: {atmos_compliance['headroom_to_18_lufs']:+.1f} dB")
    
    # Apply enhanced limiting with oversampling, lookahead, advanced envelope, and optional multiband compression
    if args.multiband:
        print(f"Processing master mix with multiband compression ({args.multiband_preset}) + enhanced limiter ({args.mode} mode, {args.lookahead}ms lookahead)...")
    else:
        print(f"Processing master mix with enhanced limiter ({args.mode} mode, {args.lookahead}ms lookahead)...")
    
    limited_audio = limit_with_oversampling(normalized_audio, config, enable_true_peak=True, 
                                          lookahead_ms=args.lookahead, limiting_mode=args.mode,
                                          multiband_preset=args.multiband_preset if args.multiband else None,
                                          multiband_enabled=args.multiband)
    
    # Apply LUFS normalization if requested
    if args.force_18_lufs:
        print("Normalizing to -18 LUFS for Atmos compliance...")
        limited_audio, applied_gain = normalize_to_lufs(limited_audio, sr, -18.0)
        print(f"Applied gain: {applied_gain:+.1f} dB")
        
        # Verify final LUFS
        final_lufs = measure_lufs(limited_audio, sr)
        print(f"Final LUFS: {final_lufs:.1f} LUFS")
    
    # Get sidechain signal using original method for comparison
    _, sidechain_signal = limit_with_sidechain(normalized_audio, config)
    
    # Generate output filenames
    master_file = "master_limited.wav"
    sidechain_file = "master_sidechain.wav"
    
    # Write master mix
    print(f"Writing limited master: {master_file}")
    sf.write(master_file, limited_audio, sr)
    
    # Write sidechain signal (mono)
    print(f"Writing sidechain signal: {sidechain_file}")
    sf.write(sidechain_file, sidechain_signal, sr)
    
    # Perform comprehensive analysis if requested
    if args.analyze:
        print("Performing comprehensive audio analysis...")
        
        # Analyze both original and limited audio
        print("Analyzing original master mix...")
        original_analysis = comprehensive_analysis(normalized_audio, sr)
        
        print("Analyzing limited master...")
        limited_analysis = comprehensive_analysis(limited_audio, sr)
        
        # Create plots
        print(f"Creating analysis plots in: {args.analysis_dir}")
        
        # Original master plots
        original_plots = create_all_plots(normalized_audio, sr, original_analysis, 
                                        f"{args.analysis_dir}/original")
        
        # Limited master plots
        limited_plots = create_all_plots(limited_audio, sr, limited_analysis, 
                                       f"{args.analysis_dir}/limited")
        
        # Print analysis summary
        print("\n" + "="*60)
        print("COMPREHENSIVE MASTER ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nORIGINAL MASTER MIX:")
        print(f"Peak: {original_analysis['basic_metrics']['peak_db']:.2f} dB")
        print(f"RMS: {original_analysis['basic_metrics']['rms_db']:.2f} dB")
        print(f"Crest Factor: {original_analysis['basic_metrics']['crest_factor']:.1f} dB")
        print(f"LUFS: {original_analysis['loudness_metrics']['lufs']:.1f} LUFS")
        print(f"LRA: {original_analysis['loudness_metrics']['lra']:.1f} LU")
        print(f"True Peak: {original_analysis['loudness_metrics']['true_peak_db']:.2f} dB")
        print(f"DR14: {original_analysis['dynamic_metrics']['dr14']:.1f} dB")
        print(f"Punch Score: {original_analysis['dynamic_metrics']['punch_metrics']['punch_score']:.1f}")
        
        print(f"\nLIMITED MASTER:")
        print(f"Peak: {limited_analysis['basic_metrics']['peak_db']:.2f} dB")
        print(f"RMS: {limited_analysis['basic_metrics']['rms_db']:.2f} dB")
        print(f"Crest Factor: {limited_analysis['basic_metrics']['crest_factor']:.1f} dB")
        print(f"LUFS: {limited_analysis['loudness_metrics']['lufs']:.1f} LUFS")
        print(f"LRA: {limited_analysis['loudness_metrics']['lra']:.1f} LU")
        print(f"True Peak: {limited_analysis['loudness_metrics']['true_peak_db']:.2f} dB")
        print(f"DR14: {limited_analysis['dynamic_metrics']['dr14']:.1f} dB")
        print(f"Punch Score: {limited_analysis['dynamic_metrics']['punch_metrics']['punch_score']:.1f}")
        
        # Stem analysis
        print(f"\nSTEM ANALYSIS:")
        print(f"Number of stems processed: {len(glob.glob(str(stems_dir / '*.wav'))) + len(glob.glob(str(stems_dir / '*.aiff'))) + len(glob.glob(str(stems_dir / '*.flac')))}")
        print(f"Total duration: {len(master_mix) / sr:.1f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Channels: {master_mix.shape[1]}")
        
        # Compliance status
        orig_compliance = original_analysis['compliance']
        limited_compliance = limited_analysis['compliance']
        
        print(f"\nCOMPLIANCE STATUS:")
        print(f"Original - Atmos Compliant: {orig_compliance['atmos_compliant']}")
        print(f"Limited - Atmos Compliant: {limited_compliance['atmos_compliant']}")
        
        print(f"\nANALYSIS PLOTS CREATED:")
        print(f"Original master plots: {len(original_plots)} files in {args.analysis_dir}/original/")
        print(f"Limited master plots: {len(limited_plots)} files in {args.analysis_dir}/limited/")
        
        for plot_type, plot_path in original_plots.items():
            print(f"  Original {plot_type}: {plot_path}")
        
        for plot_type, plot_path in limited_plots.items():
            print(f"  Limited {plot_type}: {plot_path}")
        
        print("="*60)
    
    print("Done!")
    print(f"Limited master: {master_file}")
    print(f"Sidechain signal: {sidechain_file}")


if __name__ == "__main__":
    main()