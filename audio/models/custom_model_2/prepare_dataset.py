#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Script for DTLN Training

This script takes noisy audio files from mvdr_outputs and their corresponding
clean audio files from true_labels, then splits them into train/val/test sets.

Noisy files: 1_part1_A_female_only.wav, 1_part1_B_female_music.wav, etc.
Clean files: 1_part1.flac, 99_part15.flac, etc.

The mapping extracts the base pattern (e.g., "1_part1") from noisy files
to find the corresponding clean file.

Author: Generated for DTLN-Lite training
Date: February 2026
"""

import os
import shutil
import argparse
import random
import re
from pathlib import Path
from collections import defaultdict
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm


# =============================================================================
# CONFIGURATION - Set your paths here (alternative to command line arguments)
# =============================================================================

# Path to folder containing noisy audio files (mvdr_outputs)
NOISY_FOLDER = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/post mvdr non-reverb 100k/mvdr_outputs'

# Path to folder containing clean audio files (true_labels)
CLEAN_FOLDER = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/post mvdr non-reverb 100k/true_labels'

# Output directory for prepared dataset
OUTPUT_DIR = './prepared_dataset'

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# Convert audio to 16kHz WAV format (recommended for DTLN)
# Set to False if your files are already 16kHz WAV to speed up processing
CONVERT_TO_16K = False

# Set to True to use the paths defined above instead of command line arguments
USE_CONFIG_PATHS = True

# =============================================================================


def extract_base_name(noisy_filename):
    """
    Extract the base name (e.g., '1_part1') from noisy filename.
    
    Examples:
        '1_part1_A_female_only.wav' -> '1_part1'
        '99_part15_E_female_music_noise.wav' -> '99_part15'
    """
    # Remove extension
    name = os.path.splitext(noisy_filename)[0]
    
    # Pattern: number_partNumber_rest
    # Match: digits_partDigits
    match = re.match(r'^(\d+_part\d+)', name)
    
    if match:
        return match.group(1)
    else:
        # Fallback: try to extract first two underscore-separated parts
        parts = name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return None


def find_clean_file(base_name, clean_folder, clean_files_dict):
    """
    Find the corresponding clean file for a given base name.
    
    Args:
        base_name: e.g., '1_part1'
        clean_folder: Path to true_labels folder
        clean_files_dict: Dictionary mapping base names to clean filenames
    
    Returns:
        Full path to clean file or None if not found
    """
    if base_name in clean_files_dict:
        return os.path.join(clean_folder, clean_files_dict[base_name])
    return None


def build_clean_files_dict(clean_folder):
    """
    Build a dictionary mapping base names to clean filenames.
    
    Args:
        clean_folder: Path to true_labels folder
    
    Returns:
        Dictionary: {'1_part1': '1_part1.flac', ...}
    """
    clean_dict = {}
    
    for filename in os.listdir(clean_folder):
        if filename.endswith(('.flac', '.wav', '.mp3')):
            # Extract base name from clean file (e.g., '1_part1.flac' -> '1_part1')
            base_name = os.path.splitext(filename)[0]
            clean_dict[base_name] = filename
    
    return clean_dict


def convert_to_wav_16k(input_path, output_path):
    """
    Convert audio file to 16kHz mono WAV format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
    """
    try:
        # Try using soundfile first (faster, no caching issues)
        audio, sr = sf.read(input_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy.signal.resample(audio, num_samples)
        
        # Save as WAV
        sf.write(output_path, audio.astype(np.float32), 16000)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def prepare_dataset(
    noisy_folder,
    clean_folder,
    output_base_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    convert_to_16k=True
):
    """
    Prepare dataset by splitting into train/val/test sets.
    
    Args:
        noisy_folder: Path to mvdr_outputs folder
        clean_folder: Path to true_labels folder
        output_base_dir: Base directory for output
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
        convert_to_16k: Whether to convert audio to 16kHz WAV
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_dirs = {
        'train_noisy': os.path.join(output_base_dir, 'train', 'noisy'),
        'train_clean': os.path.join(output_base_dir, 'train', 'clean'),
        'val_noisy': os.path.join(output_base_dir, 'val', 'noisy'),
        'val_clean': os.path.join(output_base_dir, 'val', 'clean'),
        'test_noisy': os.path.join(output_base_dir, 'test', 'noisy'),
        'test_clean': os.path.join(output_base_dir, 'test', 'clean'),
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Build clean files dictionary
    print("\nBuilding clean files index...")
    clean_files_dict = build_clean_files_dict(clean_folder)
    print(f"Found {len(clean_files_dict)} clean files")
    
    # Get all noisy files and find their clean counterparts
    print("\nMatching noisy files to clean files...")
    paired_files = []  # List of (noisy_path, clean_path, base_name)
    unmatched_files = []
    
    noisy_files = [f for f in os.listdir(noisy_folder) 
                   if f.endswith(('.wav', '.flac', '.mp3'))]
    
    for noisy_file in tqdm(noisy_files, desc="Matching files"):
        base_name = extract_base_name(noisy_file)
        
        if base_name is None:
            unmatched_files.append((noisy_file, "Could not extract base name"))
            continue
        
        clean_path = find_clean_file(base_name, clean_folder, clean_files_dict)
        
        if clean_path is None:
            unmatched_files.append((noisy_file, f"No clean file for base: {base_name}"))
            continue
        
        noisy_path = os.path.join(noisy_folder, noisy_file)
        paired_files.append((noisy_path, clean_path, noisy_file))
    
    print(f"\nSuccessfully matched: {len(paired_files)} file pairs")
    print(f"Unmatched files: {len(unmatched_files)}")
    
    if unmatched_files and len(unmatched_files) <= 10:
        print("\nUnmatched files:")
        for f, reason in unmatched_files:
            print(f"  - {f}: {reason}")
    elif unmatched_files:
        print(f"\nFirst 10 unmatched files:")
        for f, reason in unmatched_files[:10]:
            print(f"  - {f}: {reason}")
    
    if len(paired_files) == 0:
        print("\nERROR: No files were matched! Check your folder paths and file naming.")
        return
    
    # Shuffle and split
    print("\nShuffling and splitting dataset...")
    random.shuffle(paired_files)
    
    n_total = len(paired_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remaining goes to test
    
    train_files = paired_files[:n_train]
    val_files = paired_files[n_train:n_train + n_val]
    test_files = paired_files[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_files)} files ({len(train_files)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_files)} files ({len(val_files)/n_total*100:.1f}%)")
    print(f"  Testing:    {len(test_files)} files ({len(test_files)/n_total*100:.1f}%)")
    
    # Copy/convert files
    def process_split(files, noisy_dir, clean_dir, split_name):
        print(f"\nProcessing {split_name} set...")
        
        for noisy_path, clean_path, original_name in tqdm(files, desc=split_name):
            # Generate output filename (use same name for both noisy and clean)
            # This ensures the data generator can match them
            base_output_name = os.path.splitext(original_name)[0] + '.wav'
            
            noisy_output = os.path.join(noisy_dir, base_output_name)
            clean_output = os.path.join(clean_dir, base_output_name)
            
            if convert_to_16k:
                # Convert to 16kHz WAV
                convert_to_wav_16k(noisy_path, noisy_output)
                convert_to_wav_16k(clean_path, clean_output)
            else:
                # Just copy files
                shutil.copy2(noisy_path, noisy_output)
                # For clean files, need to handle different extensions
                if clean_path.endswith('.wav'):
                    shutil.copy2(clean_path, clean_output)
                else:
                    # Convert non-WAV clean files
                    convert_to_wav_16k(clean_path, clean_output)
    
    process_split(train_files, output_dirs['train_noisy'], output_dirs['train_clean'], 'Training')
    process_split(val_files, output_dirs['val_noisy'], output_dirs['val_clean'], 'Validation')
    process_split(test_files, output_dirs['test_noisy'], output_dirs['test_clean'], 'Testing')
    
    # Save split information
    split_info_path = os.path.join(output_base_dir, 'split_info.txt')
    with open(split_info_path, 'w') as f:
        f.write(f"Dataset Split Information\n")
        f.write(f"========================\n\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total paired files: {n_total}\n\n")
        f.write(f"Training:   {len(train_files)} files ({train_ratio*100:.0f}%)\n")
        f.write(f"Validation: {len(val_files)} files ({val_ratio*100:.0f}%)\n")
        f.write(f"Testing:    {len(test_files)} files ({test_ratio*100:.0f}%)\n\n")
        
        f.write(f"\nTraining files:\n")
        for noisy_path, clean_path, name in train_files:
            f.write(f"  {name}\n")
        
        f.write(f"\nValidation files:\n")
        for noisy_path, clean_path, name in val_files:
            f.write(f"  {name}\n")
        
        f.write(f"\nTest files:\n")
        for noisy_path, clean_path, name in test_files:
            f.write(f"  {name}\n")
    
    print(f"\nSplit information saved to: {split_info_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_base_dir}")
    print(f"\nFolder structure:")
    print(f"  {output_base_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── noisy/  ({len(train_files)} files)")
    print(f"  │   └── clean/  ({len(train_files)} files)")
    print(f"  ├── val/")
    print(f"  │   ├── noisy/  ({len(val_files)} files)")
    print(f"  │   └── clean/  ({len(val_files)} files)")
    print(f"  ├── test/")
    print(f"  │   ├── noisy/  ({len(test_files)} files)")
    print(f"  │   └── clean/  ({len(test_files)} files)")
    print(f"  └── split_info.txt")
    
    print("\n\nTo use with DTLN-Lite training, update your config:")
    print(f"  config.path_to_train_noisy = '{output_dirs['train_noisy']}'")
    print(f"  config.path_to_train_clean = '{output_dirs['train_clean']}'")
    print(f"  config.path_to_val_noisy = '{output_dirs['val_noisy']}'")
    print(f"  config.path_to_val_clean = '{output_dirs['val_clean']}'")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for DTLN training by splitting into train/val/test sets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_dataset.py -n /path/to/mvdr_outputs -c /path/to/true_labels -o /path/to/output
  
  python prepare_dataset.py \\
      --noisy_folder /path/to/mvdr_outputs \\
      --clean_folder /path/to/true_labels \\
      --output_dir /path/to/prepared_dataset \\
      --train_ratio 0.8 \\
      --val_ratio 0.1 \\
      --test_ratio 0.1 \\
      --seed 42
        """
    )
    
    parser.add_argument(
        '-n', '--noisy_folder',
        required=True,
        help='Path to folder containing noisy audio files (mvdr_outputs)'
    )
    
    parser.add_argument(
        '-c', '--clean_folder',
        required=True,
        help='Path to folder containing clean audio files (true_labels)'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Output directory for prepared dataset'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of data for validation (default: 0.1)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Ratio of data for testing (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no_convert',
        action='store_true',
        help='Skip conversion to 16kHz WAV (just copy files)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.noisy_folder):
        print(f"ERROR: Noisy folder does not exist: {args.noisy_folder}")
        return
    
    if not os.path.exists(args.clean_folder):
        print(f"ERROR: Clean folder does not exist: {args.clean_folder}")
        return
    
    # Run preparation
    prepare_dataset(
        noisy_folder=args.noisy_folder,
        clean_folder=args.clean_folder,
        output_base_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        convert_to_16k=not args.no_convert
    )


if __name__ == '__main__':
    
    # Check if using config paths or command line arguments
    if USE_CONFIG_PATHS:
        # Use paths defined at the top of the file
        print("Using paths from configuration...")
        
        # Validate paths
        if not os.path.exists(NOISY_FOLDER):
            print(f"ERROR: Noisy folder does not exist: {NOISY_FOLDER}")
            exit(1)
        
        if not os.path.exists(CLEAN_FOLDER):
            print(f"ERROR: Clean folder does not exist: {CLEAN_FOLDER}")
            exit(1)
        
        # Run preparation with config values
        prepare_dataset(
            noisy_folder=NOISY_FOLDER,
            clean_folder=CLEAN_FOLDER,
            output_base_dir=OUTPUT_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED,
            convert_to_16k=CONVERT_TO_16K
        )
    else:
        # Use command line arguments
        main()





















# User: create virtual environment and install all the packages to run the following file: python3 run_evaluation.py -i /Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/COMPARE_V4_mvdr_mic1_output.wav -o test_output -m ./pretrained_model/model.h5 

# GitHub Copilot: Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/run_evaluation.py#1-1), lines 1 to 100

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/eval_env.yml#1-1), lines 1 to 50

# Let me also check the DTLN_model.py to understand the full dependencies needed: 

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/DTLN_model.py#1-1), lines 1 to 50

# Now I have a good understanding. The main dependencies needed are:
# - TensorFlow (with Keras)
# - soundfile
# - librosa
# - numpy
# - wavinfo

# Let me create a virtual environment and install the required packages: 

# Ran terminal command:  cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master" && python3 -m venv dtln_env

# Ran terminal command:  cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master" && source dtln_env/bin/activate && pip install --upgrade pip

# Ran terminal command:  cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master" && source dtln_env/bin/activate && pip install tensorflow soundfile librosa numpy wavinfo

# All packages installed successfully. Now let's run the evaluation script: 

# Ran terminal command:  cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master" && source dtln_env/bin/activate && python3 run_evaluation.py -i "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/COMPARE_V4_mvdr_mic1_output.wav" -o test_output -m ./pretrained_model/model.h5

# The script ran successfully. The virtual environment has been created and all packages are installed. Let me verify the output was created: 

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master)

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/test_output)

# Let me check if there's an output directory in the beamforming folder or if the input was a single file: 

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/run_evaluation.py#100-100), lines 100 to 132

# I see - the script expects a **folder** for input, not a single file. The input `-i` parameter should point to a folder containing `.wav` files. Let me check if that output was created where the input file is: 

# Ran terminal command:  ls -la "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/" | head -20

# Ran terminal command:  ls -la "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master/test_output" 2>/dev/null || echo "Output folder does not exist yet"

# The script ran without processing any files because you passed a **single file** to `-i` instead of a **folder**. The script walks through a folder to find `.wav` files.

# Let me run it correctly by pointing to the folder containing the file: 

# Ran terminal command:  cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master" && source dtln_env/bin/activate && python3 run_evaluation.py -i "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output" -o test_output -m ./pretrained_model/model.h5

# All files have been processed successfully. Let me verify the output: 

# Ran terminal command:  ls -la "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master/test_output"

# Done! Here's a summary:

# ### Virtual Environment Created
# - **Location**: dtln_env in the DTLN-master folder
# - **Python version**: 3.11

# ### Packages Installed
# - TensorFlow 2.20.0
# - Keras 3.13.2
# - soundfile 0.13.1
# - librosa 0.11.0
# - numpy 2.3.5
# - wavinfo 4.0.0
# - Plus all required dependencies

# ### How to Use
# To run the evaluation script in the future, activate the virtual environment first:

# ```bash
# cd "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/DTLN-master"
# source dtln_env/bin/activate
# python3 run_evaluation.py -i <input_folder> -o <output_folder> -m ./pretrained_model/model.h5
# ```

# **Note**: The `-i` parameter expects a **folder path** (not a single file). All `.wav` files in that folder will be processed. The denoised output files are now in test_output with 10 processed audio files.

# User: i want to create a new lighter model based on this model. i have my own dataset which i'll use to retrain the new model. create a single jupyter notebook containing the whole code.

# GitHub Copilot: Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/DTLN_model.py#1-1), lines 1 to 200

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/DTLN_model.py#200-200), lines 200 to 400

# Read [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/DTLN_model.py#580-580), lines 580 to 642

# Now I have all the information needed. Let me create a comprehensive Jupyter notebook with a lighter DTLN model for training: 

# Created [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/DTLN_Lite_Training.ipynb)

# I've created a comprehensive Jupyter notebook at DTLN_Lite_Training.ipynb with everything you need to train a lighter DTLN model.

# ### Key Features of the DTLN-Lite Model:

# | Parameter | Original DTLN | DTLN-Lite | Reduction |
# |-----------|--------------|-----------|-----------|
# | LSTM Units | 128 | 64 | 50% |
# | Encoder Size | 256 | 128 | 50% |
# | Parameters | ~987K | ~275K | ~72% |

# ### Notebook Structure:

# 1. **Setup and Imports** - All required libraries
# 2. **Configuration** - Easy-to-modify parameters for model size and training
# 3. **Custom Layers** - Instant Layer Normalization
# 4. **DTLN-Lite Model Class** - Complete lighter model implementation
# 5. **Data Generator** - Loads paired noisy/clean audio files
# 6. **Build and Inspect Model** - Visualize architecture
# 7. **Training** - Full training pipeline with callbacks
# 8. **Training Visualization** - Plot loss curves
# 9. **Inference** - Process audio files and visualize results
# 10. **Save/Export** - Save as .h5, SavedModel, or TFLite
# 11. **Test with Pretrained** - Verify setup works
# 12. **Model Variants** - Pre-configured variants (Ultra-Lite, Lite, Medium)

# ### To use:

# 1. Open the notebook in VS Code or Jupyter
# 2. Update the paths in the `Config` class (cell 2) to your dataset:
#    - `path_to_train_noisy` / `path_to_train_clean`
#    - `path_to_val_noisy` / `path_to_val_clean`
# 3. Run the cells to train your model

# Your dataset should have matching filenames for noisy and clean audio (16kHz WAV format).

# Made changes.

# User: right now i have two folders, folder1 named mvdr_outputs which contrains all the noisy audios and folder2 named true_labels which contrains all the corresponding clean audios. the files in the mvdr_outputs folder have names like 1_part1_A_female_only.wav, 1_part1_B_female_music.wav..., 99_part15_E_female_music_noise.wav, .... and the corresponding clean files are in the true_labels folder with names 1_part1.flac,..,99_part15.flac,.... . Now create a python file that will take the paths of the two folders mvdr_outputs and true_labels and randomly sample 80% pecent of the data from mvdr_outputs folder for training, 10% for validation and 10% for testing, then find their corresponding clean audio (i.e., true label) from the true_labels folder. Save all the training, validation and testing files. 
# *note: do random sampling with no duplicates

# GitHub Copilot: Created [](file:///Users/emonchowdhury/Desktop/Phase%202/av_zoom/audio/models/DTLN-master/prepare_dataset.py)

# I've created prepare_dataset.py that handles your dataset preparation.

# ### How it works:

# 1. **Filename Matching**: Extracts the base pattern (e.g., `1_part1`) from noisy files like `1_part1_A_female_only.wav` and finds the corresponding clean file `1_part1.flac`

# 2. **Random Splitting**: Shuffles all matched pairs and splits into:
#    - 80% Training
#    - 10% Validation  
#    - 10% Testing

# 3. **Audio Conversion**: Converts all files to 16kHz mono WAV format (required for DTLN)

# 4. **Saves**: Creates organized folder structure with matching filenames

# ### Usage:

# ```bash
# # Activate your virtual environment first
# source dtln_env/bin/activate

# # Install tqdm if not already installed
# pip install tqdm

# # Run the script
# python prepare_dataset.py \
#     -n /path/to/mvdr_outputs \
#     -c /path/to/true_labels \
#     -o ./prepared_dataset
# ```

# ### Output Structure:
# ```
# prepared_dataset/
# ├── train/
# │   ├── noisy/  (80% of files)
# │   └── clean/  (corresponding clean files)
# ├── val/
# │   ├── noisy/  (10% of files)
# │   └── clean/  (corresponding clean files)
# ├── test/
# │   ├── noisy/  (10% of files)
# │   └── clean/  (corresponding clean files)
# └── split_info.txt  (log of which files went where)
# ```

# ### Then update your notebook config:
# ```python
# config.path_to_train_noisy = './prepared_dataset/train/noisy/'
# config.path_to_train_clean = './prepared_dataset/train/clean/'
# config.path_to_val_noisy = './prepared_dataset/val/noisy/'
# config.path_to_val_clean = './prepared_dataset/val/clean/'
# ```

# Made changes.

# User: add option to set the paths in the beginning of the code not just by passing as argument 

# GitHub Copilot: Using "Replace String in File"


# ```

# ```


# Using "Replace String in File"


# ```

# ```
# Done! I've added a configuration section at the top of prepare_dataset.py. 

# ### How to use:

# **Option 1: Set paths in the code**
# Edit the configuration section at the top of the file:
# ```python
# # Path to folder containing noisy audio files (mvdr_outputs)
# NOISY_FOLDER = '/path/to/mvdr_outputs'

# # Path to folder containing clean audio files (true_labels)
# CLEAN_FOLDER = '/path/to/true_labels'

# # Output directory for prepared dataset
# OUTPUT_DIR = './prepared_dataset'

# # Set to True to use the paths defined above
# USE_CONFIG_PATHS = True  # <-- Change this to True
# ```

# Then simply run:
# ```bash
# python prepare_dataset.py
# ```

# **Option 2: Use command line arguments (original method)**
# Keep `USE_CONFIG_PATHS = False` and run:
# ```bash
# python prepare_dataset.py -n /path/to/mvdr_outputs -c /path/to/true_labels -o ./prepared_dataset
# ```

# Made changes.

# User: is there any benifit of having the option of passing the paths as arguments

# GitHub Copilot: There are a few benefits to keeping command line arguments:

# **Pros of command line arguments:**
# - **Scripting/Automation**: Can call from bash scripts without modifying code
# - **Multiple runs**: Easy to process different folders without editing the file
# - **No accidental edits**: Don't risk breaking the code by editing it

# **However**, for your use case (personal project, likely one-time dataset prep), these benefits are minimal.

# If you prefer a simpler file with just the inline configuration, I can remove the argument parsing entirely. Would you like me to simplify it to just use the config section at the top?

# User: my dataset has samples which are all 3s in length which are outputs from MVDR. the outputs are 16kHz audios and the mvdr used the following STFT parameters:
# % -------- STFT params --------
# N = 256;
# hop = 128;
# nfft = 512;
# window = sqrt(hann(N,'periodic'));

# is there any need to modify the current model ?


# GitHub Copilot: Great question! Let me analyze the compatibility:

# ### Your Setup vs DTLN Parameters:

# | Parameter | Your MVDR | DTLN Default | DTLN-Lite |
# |-----------|-----------|--------------|-----------|
# | Sample Rate | 16kHz ✅ | 16kHz | 16kHz |
# | Hop Size | 128 ✅ | 128 | 128 |
# | Window/Block | 256 | 512 | 512 |
# | FFT Size | 512 | 512 | 512 |

# ### Key Points:

# 1. **No architectural changes needed** - DTLN processes **raw time-domain audio**, not STFT features. It applies its own internal STFT, so your MVDR's STFT parameters don't need to match.

# 2. **Sample rate matches** ✅ - Both use 16kHz

# 3. **Main change needed: Audio chunk length** - Your samples are 3 seconds, but the default config uses 15 seconds. You need to update this.

# ### Required Config Change:

# In your notebook, update the `Config` class:

# ```python
# # Training parameters
# len_samples = 3    # Changed from 15 to 3 seconds (your audio length)
# ```

# Or if you want to be precise (accounting for block alignment):
# ```python
# # 3 seconds at 16kHz = 48,000 samples
# # Aligned to block_shift (128): floor(48000/128)*128 = 47,872 samples
# len_samples = 47872 / 16000  # ≈ 2.992 seconds
# ```

# ### Optional Optimization:

# If you want DTLN to use the same window size as your MVDR (256 instead of 512), you can change:
# ```python
# block_len = 256    # Match MVDR window (optional)
# ```

# This would give you 129 frequency bins instead of 257, making the model even lighter. But it's **not required** - the default 512 will work fine.

# **Bottom line**: Just change `len_samples = 3` in the config and you're good to go!