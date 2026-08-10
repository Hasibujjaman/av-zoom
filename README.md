# av-zoom

Audio-Visual Zooming — MATLAB implementations for audio beamforming, DOA estimation, STFT utilities, simulation scenarios, and fusion/evaluation tools for zooming systems.

## Highlights
- MATLAB-based toolbox and experiments for audio processing relevant to audio-visual zooming.
- Components for beamforming (MVDR, delay-and-sum), DOA estimation, STFT/ISTFT utilities, simulation scenarios (anechoic/reverberant), fusion of AV cues, and evaluation/plotting scripts.
- Intended for researchers and engineers working on audio-visual enhancement, beamforming, and source localization.

## Repository structure
Top-level layout (annotated):
```
audio/                  audio processing utilities
  beamforming/          MVDR, delay-and-sum, STFT wrappers, test signals, dataset generation, outputs
  doa/                  direction-of-arrival (DOA) estimation algorithms and tests (gcc_phat, tracking)
  stft/                 compute_stft / compute_istft and STFT roundtrip tests
fusion/                 audio-visual fusion helpers (av_doa_fusion.m, confidence metrics)
simulation/             example scenarios and scripts (task1_anechoic.m, task2_reverb.m)
evaluation/             metrics and plotting scripts (metrics.m, plots.m)
FIGURES/                supporting figures used in reports/papers
sync/                   (project sync utilities / scripts)
README.md               this file
.gitignore
```

How it fits together:
- Simulation scripts generate or load mixtures (simulation/* and audio/beamforming/create_*.m), STFT utilities convert between time/frequency domains (audio/stft/), DOA routines estimate source direction (audio/doa/), beamforming uses DOA and covariance routines to compute and apply spatial filters (audio/beamforming/), fusion modules combine AV estimates (fusion/), and evaluation scripts measure and plot results (evaluation/).

## Requirements
- MATLAB (R2018b or later recommended)
- Signal Processing Toolbox (for STFT, filtering helpers)
- Optional: Audio Toolbox (if using advanced audio I/O), but basic MATLAB functions suffice for the provided scripts.

## Quickstart (run a simple simulation)
1. Clone the repository:
   ```bash
   git clone git@github.com:Hasibujjaman/av-zoom.git
   cd av-zoom
   ```
2. Start MATLAB and add the project path:
   ```matlab
   addpath(genpath(pwd));
   ```
3. Run a provided simulation:
   - Anechoic task:
     ```matlab
     simulation/task1_anechoic
     ```
   - Reverberant task:
     ```matlab
     simulation/task2_reverb
     ```
4. Try beamforming and reconstruction:
   ```matlab
   % example: compute MVDR weights and reconstruct audio
   audio/beamforming/create_mixture;         % prepare test mixture (or run provided examples)
   weights = audio/beamforming/compute_mvdr_weights(...); % see function signature
   audio/beamforming/reconstruct_mvdr_audio(...);
   ```
5. Run DOA example:
   ```matlab
   audio/doa/test_doa_tracking
   ```
6. Run STFT roundtrip test:
   ```matlab
   audio/stft/test_stft_roundtrip
   ```
7. Evaluate results:
   ```matlab
   evaluation/metrics
   evaluation/plots
   ```

(For function arguments and options, open the corresponding .m files — most functions include short in-file comments describing inputs/outputs.)

## Notable files
- audio/beamforming/create_reverberant_mixture.m — dataset / mixture generation (reverberant)
- audio/beamforming/compute_mvdr_weights.m — MVDR beamformer weight computation
- audio/doa/estimate_doa.m, gcc_phat.m — DOA estimation primitives
- audio/stft/compute_stft.m, compute_istft.m — STFT/ISTFT helpers
- fusion/av_doa_fusion.m, fusion/confidence_metrics.m — AV fusion and scoring
- simulation/task1_anechoic.m, simulation/task2_reverb.m — ready-to-run experiment scripts
- evaluation/metrics.m, evaluation/plots.m — evaluation & visualization

## Usage tips
- Before running experiments, open the simulation script to confirm paths and variable names for input audio or test signals.
- Many helper scripts assume relative project paths; use `addpath(genpath(pwd))` at the project root to ensure MATLAB can find all functions.
- For reproducible results, set random seeds where appropriate in your own experiments.

## Contributing
- Issues and pull requests are welcome. Please include a concise description of the change and any supporting figures or example scripts.
- If adding new datasets or large binary files (audio), use an external data host or Git LFS.

## License & Citation
- Add a LICENSE file if you want to specify reuse terms. If this code accompanies a paper, include citation instructions here.

## Contact
- For questions or suggestions, open an issue on the repository or contact the maintainer (repository owner).
