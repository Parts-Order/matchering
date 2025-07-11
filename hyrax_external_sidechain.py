#!/usr/bin/env python3
"""
Hyrax External Sidechain Limiter

Applies Hyrax limiter to one audio file using a separate audio file as the sidechain trigger.

Usage: python hyrax_external_sidechain.py <input_audio> <sidechain_audio>
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import math
from scipy import signal
from scipy.ndimage import maximum_filter1d

# Import matchering components
import matchering as mg
from matchering.defaults import Config
from matchering.dsp import rectify, flip, max_mix, normalize
from matchering.utils import make_odd, ms_to_samples
from matchering.limiter.enhanced import limit_with_external_sidechain_oversampling, get_limiter_analysis
from matchering.lufs import measure_lufs, check_atmos_compliance, normalize_to_lufs


def limit_with_external_sidechain(input_audio: np.ndarray, sidechain_audio: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified Hyrax limiter that uses external sidechain for gain reduction calculation
    but applies the limiting to the input audio.
    """
    
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
    
    # Use SIDECHAIN audio for gain reduction calculation
    rectified = rectify(sidechain_audio, config.threshold)
    
    print(f"DEBUG: Sidechain peak: {np.abs(sidechain_audio).max():.6f}")
    print(f"DEBUG: Threshold: {config.threshold:.6f}")
    print(f"DEBUG: Rectified range: {rectified.min():.6f} to {rectified.max():.6f}")
    print(f"DEBUG: Rectified unique values: {len(np.unique(rectified))}")
    
    if np.all(np.isclose(rectified, 1.0)):
        print("DEBUG: No limiting needed - rectified is all 1.0")
        gain_envelope = np.ones(input_audio.shape[0])
        return input_audio, gain_envelope
    
    # Calculate gain envelope from SIDECHAIN audio
    gain_hard_clip = flip(1.0 / rectified)
    print(f"DEBUG: Gain hard clip range: {gain_hard_clip.min():.6f} to {gain_hard_clip.max():.6f}")
    
    gain_attack, gain_hard_clip_slided = __process_attack(np.copy(gain_hard_clip), config)
    print(f"DEBUG: Gain attack range: {gain_attack.min():.6f} to {gain_attack.max():.6f}")
    
    gain_release = __process_release(np.copy(gain_hard_clip_slided), config)
    print(f"DEBUG: Gain release range: {gain_release.min():.6f} to {gain_release.max():.6f}")
    
    gain = flip(max_mix(gain_hard_clip, gain_attack, gain_release))
    print(f"DEBUG: Final gain range: {gain.min():.6f} to {gain.max():.6f}")
    
    # Clamp gain to prevent explosions
    gain = np.clip(gain, 0.0, 1.0)
    
    # Apply gain envelope to INPUT audio (not sidechain)
    limited_audio = input_audio * gain[:, None]
    
    # Verify no sidechain bleed
    print(f"DEBUG: Input audio range: {input_audio.min():.6f} to {input_audio.max():.6f}")
    print(f"DEBUG: Sidechain audio range: {sidechain_audio.min():.6f} to {sidechain_audio.max():.6f}")
    print(f"DEBUG: Limited audio range: {limited_audio.min():.6f} to {limited_audio.max():.6f}")
    print(f"DEBUG: Limited audio should NOT match sidechain range")
    
    # Return the RAW GAIN as sidechain signal (not inverted)
    sidechain_signal = gain
    
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
    import glob
    audio_extensions = ['*.wav', '*.aiff', '*.flac', '*.mp3']
    stem_files = []
    for ext in audio_extensions:
        stem_files.extend(glob.glob(str(stems_dir / ext)))
    
    if not stem_files:
        raise ValueError(f"No audio files found in {stems_dir}")
    
    print(f"Found {len(stem_files)} stems for summing:")
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
    
    print(f"Summed to create sidechain: {master_mix.shape[0]} samples, {master_mix.shape[1]} channels")
    return master_mix, sample_rate


def match_audio_specs(audio1: np.ndarray, audio2: np.ndarray, sr1: int, sr2: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Match sample rates, lengths, and channel counts between two audio files.
    """
    # Check sample rates
    if sr1 != sr2:
        print(f"Error: Sample rate mismatch ({sr1} vs {sr2})")
        print("Both files must have the same sample rate")
        sys.exit(1)
    
    # Match lengths (pad shorter with zeros)
    if audio1.shape[0] != audio2.shape[0]:
        max_len = max(audio1.shape[0], audio2.shape[0])
        if audio1.shape[0] < max_len:
            padding = max_len - audio1.shape[0]
            audio1 = np.pad(audio1, ((0, padding), (0, 0)), mode='constant')
        if audio2.shape[0] < max_len:
            padding = max_len - audio2.shape[0]
            audio2 = np.pad(audio2, ((0, padding), (0, 0)), mode='constant')
    
    # Match channel counts
    if audio1.shape[1] != audio2.shape[1]:
        if audio1.shape[1] == 1 and audio2.shape[1] == 2:
            audio1 = np.repeat(audio1, 2, axis=1)
        elif audio1.shape[1] == 2 and audio2.shape[1] == 1:
            audio2 = np.repeat(audio2, 2, axis=1)
        else:
            print(f"Error: Channel count mismatch ({audio1.shape[1]} vs {audio2.shape[1]})")
            sys.exit(1)
    
    return audio1, audio2, sr1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyrax External Sidechain Limiter with LUFS compliance")
    parser.add_argument("sidechain_audio", help="The audio file that triggers the limiting OR directory of stems when using --sum-stems-for-sidechain")
    parser.add_argument("input_audio", nargs='?', help="The audio file to be limited (not needed when using --process-stems with --sum-stems-for-sidechain)")
    parser.add_argument("--process-stems", action="store_true",
                       help="Process each stem file individually using the same sidechain")
    parser.add_argument("--sum-stems-for-sidechain", action="store_true",
                       help="Create sidechain by summing all stems in the directory (use with --process-stems)")
    parser.add_argument("--force-18-lufs", action="store_true", 
                       help="Force output to -18 LUFS for Atmos compliance")
    parser.add_argument("--lookahead", type=float, default=5.0,
                       help="Lookahead buffer in milliseconds (0-20ms, default: 5.0)")
    parser.add_argument("--mode", choices=["transparent", "punchy", "aggressive"], default="transparent",
                       help="Limiting mode: transparent (smooth), punchy (fast transients), aggressive (tight control)")
    parser.add_argument("--multiband", action="store_true",
                       help="Enable multiband compression before limiting")
    parser.add_argument("--multiband-preset", choices=["default", "3band", "broadcast", "mastering"], 
                       default="default", help="Multiband compressor preset (default: default)")
    parser.add_argument("--stems-mode", action="store_true",
                       help="Optimize limiter settings for individual stems (higher threshold, gentler processing)")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.sum_stems_for_sidechain and not args.process_stems:
        print("Error: --sum-stems-for-sidechain requires --process-stems")
        sys.exit(1)
    
    if args.process_stems and args.sum_stems_for_sidechain:
        # Both stem processing and stem summing for sidechain
        # In this case, the first argument (sidechain_audio) is actually the stems directory
        stems_path = Path(args.sidechain_audio)
        if not stems_path.exists() or not stems_path.is_dir():
            print(f"Error: Stems directory '{stems_path}' not found or is not a directory")
            sys.exit(1)
    elif args.process_stems:
        # Regular stem processing with external sidechain
        if not args.input_audio:
            print("Error: input_audio is required when using --process-stems without --sum-stems-for-sidechain")
            sys.exit(1)
        stems_path = Path(args.input_audio)
        sidechain_file = Path(args.sidechain_audio)
        if not stems_path.exists() or not stems_path.is_dir():
            print(f"Error: Stems directory '{stems_path}' not found or is not a directory")
            sys.exit(1)
        if not sidechain_file.exists():
            print(f"Error: Sidechain file '{sidechain_file}' not found")
            sys.exit(1)
    else:
        # Single file processing
        if not args.input_audio:
            print("Error: input_audio is required for single file processing")
            sys.exit(1)
        input_file = Path(args.input_audio)
        sidechain_file = Path(args.sidechain_audio)
        if not input_file.exists():
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)
        if not sidechain_file.exists():
            print(f"Error: Sidechain file '{sidechain_file}' not found")
            sys.exit(1)
    
    # Handle sidechain creation
    if args.sum_stems_for_sidechain:
        # Create sidechain by summing stems
        print("Creating sidechain by summing all stems...")
        try:
            sidechain_audio, sidechain_sr = sum_stems(stems_path)
        except Exception as e:
            print(f"Error creating sidechain from stems: {e}")
            sys.exit(1)
        
        # Find all audio files for individual processing
        import glob
        audio_extensions = ['*.wav', '*.aiff', '*.flac', '*.mp3']
        stem_files = []
        for ext in audio_extensions:
            stem_files.extend(glob.glob(str(stems_path / ext)))
        
        print(f"Found {len(stem_files)} stem files to process individually:")
        for stem in stem_files:
            print(f"  - {Path(stem).name}")
    
    elif args.process_stems:
        # Load external sidechain for stem processing
        print(f"Loading external sidechain: {sidechain_file.name}")
        try:
            sidechain_audio, sidechain_sr = sf.read(str(sidechain_file))
            if len(sidechain_audio.shape) == 1:
                sidechain_audio = sidechain_audio.reshape(-1, 1)
        except Exception as e:
            print(f"Error loading sidechain audio: {e}")
            sys.exit(1)
        
        # Find all audio files in the directory
        import glob
        audio_extensions = ['*.wav', '*.aiff', '*.flac', '*.mp3']
        stem_files = []
        for ext in audio_extensions:
            stem_files.extend(glob.glob(str(stems_path / ext)))
        
        if not stem_files:
            print(f"Error: No audio files found in '{stems_path}'")
            sys.exit(1)
        
        print(f"Found {len(stem_files)} stem files to process:")
        for stem in stem_files:
            print(f"  - {Path(stem).name}")
    
    else:
        # Single file processing - load sidechain
        print(f"Loading sidechain audio: {sidechain_file.name}")
        try:
            sidechain_audio, sidechain_sr = sf.read(str(sidechain_file))
            if len(sidechain_audio.shape) == 1:
                sidechain_audio = sidechain_audio.reshape(-1, 1)
        except Exception as e:
            print(f"Error loading sidechain audio: {e}")
            sys.exit(1)
        
        stem_files = [str(input_file)]
    
    print(f"Sidechain ready: {sidechain_audio.shape[0]} samples, {sidechain_audio.shape[1]} channels, {sidechain_sr}Hz")
    
    # Process each stem file
    processed_files = []
    
    for i, stem_file in enumerate(stem_files):
        print(f"\n{'='*60}")
        print(f"Processing stem {i+1}/{len(stem_files)}: {Path(stem_file).name}")
        print(f"{'='*60}")
        
        # Load input audio
        print(f"Loading input audio: {Path(stem_file).name}")
        try:
            input_audio, input_sr = sf.read(str(stem_file))
            if len(input_audio.shape) == 1:
                input_audio = input_audio.reshape(-1, 1)
        except Exception as e:
            print(f"Error loading input audio: {e}")
            continue
        
        # Match audio specifications
        input_audio, matched_sidechain, sr = match_audio_specs(input_audio, sidechain_audio, input_sr, sidechain_sr)
        
        print(f"Processing: {input_audio.shape[0]} samples, {input_audio.shape[1]} channels, {sr}Hz")
        
        # Create config with appropriate limiter settings
        config = Config()
        config.internal_sample_rate = sr
        
        if args.stems_mode:
            # STEMS MODE: Gentler settings for individual stem processing
            config.threshold = 0.85  # Much higher threshold (~-1.4dBFS) - only catch peaks
            # Adjust attack/release for stems (faster, less "glue")
            config.limiter.attack = config.limiter.attack * 0.7  # Slightly faster attack
            config.limiter.release = config.limiter.release * 0.8  # Faster release
            print(f"STEMS MODE: Using gentler limiting (threshold: {config.threshold:.2f}, ~{20*np.log10(config.threshold):.1f}dBFS)")
        else:
            # MASTER MODE: Aggressive settings for master bus limiting
            config.threshold = 0.5  # Lower threshold to ensure sidechain triggers
            print(f"MASTER MODE: Using aggressive limiting (threshold: {config.threshold:.2f}, ~{20*np.log10(config.threshold):.1f}dBFS)")
        
        # Normalize both audio files
        normalized_input, _ = normalize(input_audio, config.threshold, config.min_value, False)
        normalized_sidechain, _ = normalize(matched_sidechain, config.threshold, config.min_value, False)
        
        # Analyze audio before processing
        print("Analyzing input audio...")
        analysis = get_limiter_analysis(normalized_input, config)
        print(f"Peak: {analysis['peak_db']:.2f}dB, True Peak: {analysis['true_peak_db']:.2f}dB")
        print(f"Crest Factor: {analysis['crest_factor']:.1f}dB")
        print(f"Needs oversampling: {analysis['needs_oversampling']}")
        print(f"EBU R128 compliant: {analysis['ebu_compliant']}")
        
        # LUFS analysis
        print("Analyzing LUFS...")
        atmos_compliance = check_atmos_compliance(normalized_input, sr)
        print(f"LUFS: {atmos_compliance['lufs']:.1f} LUFS")
        print(f"LRA: {atmos_compliance['lra']:.1f} LU")
        print(f"Atmos compliant: {atmos_compliance['atmos_compliant']}")
        print(f"Headroom to -18 LUFS: {atmos_compliance['headroom_to_18_lufs']:+.1f} dB")
        
        # Determine multiband preset based on stems mode
        if args.multiband and args.stems_mode:
            # Force mastering preset for stems mode (gentlest)
            effective_preset = "mastering"
            print(f"STEMS MODE: Auto-selected 'mastering' multiband preset for gentler stem processing")
        elif args.multiband:
            effective_preset = args.multiband_preset
        else:
            effective_preset = None
        
        # Apply enhanced limiting with external sidechain, lookahead, advanced envelope, and optional multiband compression
        if args.multiband:
            print(f"Processing with multiband compression ({effective_preset}) + enhanced external sidechain ({args.mode} mode, {args.lookahead}ms lookahead)...")
        else:
            print(f"Processing with enhanced external sidechain ({args.mode} mode, {args.lookahead}ms lookahead)...")
        
        limited_audio, sidechain_signal = limit_with_external_sidechain_oversampling(
            normalized_input, normalized_sidechain, config, enable_true_peak=True, 
            lookahead_ms=args.lookahead, limiting_mode=args.mode,
            multiband_preset=effective_preset,
            multiband_enabled=args.multiband
        )
        
        # Apply LUFS normalization if requested
        if args.force_18_lufs:
            print("Normalizing to -18 LUFS for Atmos compliance...")
            limited_audio, applied_gain = normalize_to_lufs(limited_audio, sr, -18.0)
            print(f"Applied gain: {applied_gain:+.1f} dB")
            
            # Verify final LUFS
            final_lufs = measure_lufs(limited_audio, sr)
            print(f"Final LUFS: {final_lufs:.1f} LUFS")
        
        # Generate output filenames
        input_base = Path(stem_file).stem
        if args.sum_stems_for_sidechain:
            # Using summed stems as sidechain
            limited_file = f"{input_base}_limited_by_summed_stems.wav"
            envelope_file = f"{input_base}_envelope_from_summed_stems.wav"
        else:
            # Using external sidechain file
            sidechain_base = sidechain_file.stem
            limited_file = f"{input_base}_limited_by_{sidechain_base}.wav"
            envelope_file = f"{input_base}_envelope_from_{sidechain_base}.wav"
        
        # Write limited audio
        print(f"Writing limited audio: {limited_file}")
        sf.write(limited_file, limited_audio, sr)
        
        # Write sidechain envelope
        print(f"Writing sidechain envelope: {envelope_file}")
        sf.write(envelope_file, sidechain_signal, sr)
        
        processed_files.append((limited_file, envelope_file))
        print(f"✓ Completed: {Path(stem_file).name}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    if args.sum_stems_for_sidechain:
        print(f"Processed {len(processed_files)} stem files using summed stems as sidechain")
    elif args.process_stems:
        print(f"Processed {len(processed_files)} stem files using external sidechain: {sidechain_file.name}")
    else:
        print(f"Processed single file using external sidechain: {sidechain_file.name}")
    
    print(f"Files created:")
    for limited_file, envelope_file in processed_files:
        print(f"  Limited: {limited_file}")
        print(f"  Envelope: {envelope_file}")
    
    print("Done!")


if __name__ == "__main__":
    main()