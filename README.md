# av-zoom

Audio-Visual Zooming — MATLAB + Python research code

This repository implements an audio-visual zooming research pipeline: using audio-based localization and speech/enhancement models together with video processing to produce an automatically zoomed video crop that follows active sound sources (for example, a speaking person). The codebase is primarily MATLAB for the beamforming, simulation, and evaluation pipeline, with Python (PyTorch) models for neural enhancement/post-processing.

Table of contents
- About
- Repository layout
- Requirements
- Quick start
- MATLAB pipeline: run & examples
- Python models: training & inference (CRN_Light, DeepFilterNet)
- Data, evaluation & expected outputs
- Configuration
- Notes & known issues
- Contributing
- License
- Contact

About
-------
The goal of av-zoom is to fuse audio localization and single/multi-channel audio enhancement with visual cropping and tracking to create zoomed videos centered on active sound sources. The repository contains:
- MATLAB scripts for beamforming, simulation, dataset generation, and evaluation.
- Python model definitions, training notebooks, and utilities for audio enhancement (e.g., CRN_Light and DeepFilterNet variants).
- Example data folders (audio/, simulation/, evaluation/), figure assets, and a paper/ directory containing related writeups.

Repository layout
-----------------
Top-level (selected):
- audio/                       MATLAB beamforming, dataset generation, comparisons, and model outputs
  - beamforming/               MATLAB beamforming and evaluation scripts
  - models/                    Python model code, notebooks, checkpoints directories, enhanced outputs
    - custom_model_1/          CRN_Light model, training/inference notebooks and utilities
    - custom_model_2/          DeepFilterNet notebooks and outputs
- FIGURES/                     figures used in paper/demos
- simulation/                  scripts for simulated room impulse responses and mixtures
- evaluation/                  evaluation scripts and example outputs
- paper/                       paper drafts and supplementary materials

Requirements
------------
MATLAB side:
- MATLAB (R2018b or newer recommended)
- Optional toolboxes (used by some scripts/features): Signal Processing Toolbox, Audio Toolbox, Image Processing Toolbox, Computer Vision Toolbox

Python/model side (for training or running enhancements):
- Python 3.8+ (Anaconda recommended)
- PyTorch (see audio/models/custom_model_1/requirements.txt for exact pinned versions)
- Common Python packages: numpy, scipy, soundfile, librosa (check requirements.txt in each model folder)

Quick start
-----------
1. Clone the repository:

   git clone https://github.com/Hasibujjaman/av-zoom.git
   cd av-zoom

2. MATLAB: add repository to MATLAB path (from MATLAB):

   addpath(genpath(pwd));

3. Inspect demos and example scripts under `audio/beamforming/` and `simulation/`.
   - Typical demo scripts are named like `demo_*.m` or live/test scripts under `audio/beamforming/COMPARISONS & TESTS/`.

4. Python models: create a Python environment and install model deps (example for custom_model_1):

   cd audio/models/custom_model_1
   python -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt

5. Run a training notebook or `train_crn.py` from the `custom_model_1` folder to reproduce experiments. See the included Jupyter notebooks for example training and inference flows.

MATLAB pipeline: run & examples
------------------------------
- The `audio/beamforming` folder contains dataset generation, mixture creation, MVDR beamformer tests, and comparison scripts.
- Example workflow in MATLAB:
  1. Generate or load mixtures (see `create_mixture.m` and the `Dataset Generation/` scripts).
  2. Run beamformer and compute STFT-based processing (STFT params are defined in test scripts).
  3. Save enhanced outputs (WAV) into a folder such as `audio/models/.../model_enhanced_output` to be used by comparison/evaluation scripts.
- Many scripts include plotting and diagnostic steps (spectrograms, RIR plots, metric computations) — run interactively for best results.

Python models: training & inference
----------------------------------
- custom_model_1 (CRN_Light):
  - `crn_light.py` defines a compact CRN-like U-Net for post-enhancement (PyTorch). The project includes `train_crn.py`, `dataset.py`, `stft_utils.py`, loss functions and training notebooks (`.ipynb`).
  - Use `requirements.txt` in the model folder to setup the Python environment.
  - Notebooks show example preprocessing, training loops, and how to write enhanced audio outputs to `model_enhanced_output/`.

- custom_model_2 (DeepFilterNet variants):
  - Training notebooks and a `prepare_dataset.py` script are provided. Check `audio/models/custom_model_2/` for details and example outputs.

Inference notes:
- If you want MATLAB scripts to consume model outputs, save enhanced audio (WAV) to a path expected by the MATLAB comparisons (e.g., `audio/models/custom_model_1/model_enhanced_output/`). Many MATLAB scripts use `audioread()` on files placed there.
- Several MATLAB scripts currently contain absolute local file paths (e.g., `/Users/emonchowdhury/Desktop/Phase 2/av_zoom/...`). Before running those scripts on a different machine, change those paths to relative paths or set a configuration variable (see Notes & known issues below).

Data, evaluation & expected outputs
----------------------------------
- Generated mixtures and RIRs: `audio/beamforming/Dataset Generation/` creates reverberant mixtures and room impulse responses (ISM shoebox models and custom RIRs).
- Evaluation scripts compute metrics such as SI-SDR, STOI, OSINR, and use ViSQOL/MOS where available. See `audio/beamforming/COMPARISONS & TESTS/` for examples.
- Typical outputs: WAV files for enhanced signals, diagnostic plots, zoomed video outputs (if video modules are used), and CSV/log files for metric summaries.

Configuration
-------------
- Many MATLAB scripts define configuration variables at the top of the file (zoomFactor, cropSize, STFT parameters, localization method, etc.).
- Python training runs are configured either via notebooks or top-of-script config variables (training hyperparameters, dataset paths).
- Recommended: create a small `config.m` (MATLAB) or `config.yaml` (Python) and update scripts to read from these centralized configs to avoid editing many files.

Notes & known issues
--------------------
- Hardcoded absolute paths: Several MATLAB and comparison scripts use hardcoded absolute paths to user directories. These will fail on other machines — search for `/Users/` and change to repository-relative paths or parametrize.
- Pretrained weights: The repository contains training notebooks and checkpoint directories but does not include large pretrained model weights committed as binaries. If you want to share weights, consider storing them externally (release, cloud storage) and add download instructions in this README.
- License: No LICENSE file is present in this repository. Add a LICENSE file (for example MIT) if you wish to permit reuse.

Contributing
------------
Contributions are welcome. Typical ways to contribute:
- Open an issue to discuss bugs or feature requests.
- Create pull requests for small fixes (path fixes, documentation, examples) and include clear descriptions and example inputs/outputs when relevant.
- If adding pretrained model files, prefer releasing them via GitHub Releases or external storage and provide download scripts.

Suggested immediate improvements
- Replace absolute local paths with relative paths or a config variable.
- Add a short inference example script in `audio/models/custom_model_1/` that runs `crn_light.py` on a sample mixture and writes WAV output into `model_enhanced_output/` so MATLAB evaluation scripts can consume it easily.
- Commit a LICENSE file if you want to clarify reuse terms.

License
-------
No license has been specified. If you want to enable reuse, add a LICENSE file (for example, the MIT License).

Contact
-------
Maintainer: Hasibujjaman
Repository: https://github.com/Hasibujjaman/av-zoom


