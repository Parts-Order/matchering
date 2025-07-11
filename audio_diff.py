#!/usr/bin/env python3
"""
Audio Diff Tool

Creates a difference signal between two audio files.

Usage: python audio_diff.py <file1> <file2>
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python audio_diff.py <file1> <file2>")
        sys.exit(1)
    
    file1_path = Path(sys.argv[1])
    file2_path = Path(sys.argv[2])
    
    if not file1_path.exists():
        print(f"Error: File '{file1_path}' not found")
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"Error: File '{file2_path}' not found")
        sys.exit(1)
    
    # Load audio files
    print(f"Loading: {file1_path.name}")
    try:
        audio1, sr1 = sf.read(str(file1_path))
        if len(audio1.shape) == 1:
            audio1 = audio1.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading {file1_path}: {e}")
        sys.exit(1)
    
    print(f"Loading: {file2_path.name}")
    try:
        audio2, sr2 = sf.read(str(file2_path))
        if len(audio2.shape) == 1:
            audio2 = audio2.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading {file2_path}: {e}")
        sys.exit(1)
    
    # Check sample rates match
    if sr1 != sr2:
        print(f"Error: Sample rate mismatch ({sr1} vs {sr2})")
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
    
    # Calculate difference
    print("Calculating difference...")
    diff = audio1 - audio2
    
    # Generate output filename
    diff_file = f"{file1_path.stem}_minus_{file2_path.stem}_diff.wav"
    
    # Write difference file
    print(f"Writing difference: {diff_file}")
    sf.write(diff_file, diff, sr1)
    
    # Print some stats
    max_diff = np.abs(diff).max()
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"Difference file: {diff_file}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"RMS difference: {rms_diff:.6f}")


if __name__ == "__main__":
    main()