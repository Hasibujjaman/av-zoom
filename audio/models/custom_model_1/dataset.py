import os
import torch
import numpy as np

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ModuleNotFoundError:
    torchaudio = None
    _HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ModuleNotFoundError:
    sf = None
    _HAS_SOUNDFILE = False
from torch.utils.data import Dataset


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    # Prefer soundfile when available: it's lightweight and avoids optional torchaudio decoder deps.
    if _HAS_SOUNDFILE:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [N, C]
        if sr != target_sr:
            if _HAS_TORCHAUDIO:
                wav = torch.from_numpy(np.asarray(data).T)  # [C, N]
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr = target_sr
            else:
                raise ValueError(
                    f"Sample rate mismatch for {path}: got {sr}, expected {target_sr}. "
                    "Install torchaudio to enable resampling."
                )
        else:
            wav = torch.from_numpy(np.asarray(data).T)  # [C, N]
    elif _HAS_TORCHAUDIO:
        try:
            wav, sr = torchaudio.load(path)  # [C, N]
        except Exception as e:
            raise ImportError(
                "Failed to load audio via torchaudio. If you're on a minimal install, you may need "
                "an additional backend (e.g., `torchcodec`) or install `soundfile` to load wavs."
            ) from e

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        wav = wav.to(torch.float32)
    else:
        raise ModuleNotFoundError("Missing audio backend. Install `soundfile` or `torchaudio`.")

    if wav.dim() != 2:
        raise RuntimeError(f"Expected waveform shape [C,N], got {tuple(wav.shape)} for {path}")

    # Mix to mono if multi-channel
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    return wav

class MVDRDataset(Dataset):
    def __init__(self, mvdr_dir, clean_dir, sample_rate=16000):
        self.mvdr_dir = mvdr_dir
        self.clean_dir = clean_dir
        self.sr = sample_rate

        self.mvdr_files = sorted([
            f for f in os.listdir(mvdr_dir)
            if f.endswith(".wav")
        ])

        # Index clean references by stem (supports .wav/.flac)
        self.clean_index = {}
        for f in os.listdir(clean_dir):
            lower = f.lower()
            if not (lower.endswith(".wav") or lower.endswith(".flac")):
                continue
            stem, _ = os.path.splitext(f)
            self.clean_index[stem] = os.path.join(clean_dir, f)

    def _get_clean_stem(self, mvdr_name):
        # 1_part13_A_female_only.wav → 1_part13
        return "_".join(mvdr_name.split("_")[:2])

    def __len__(self):
        return len(self.mvdr_files)

    def __getitem__(self, idx):
        mvdr_name = self.mvdr_files[idx]
        clean_stem = self._get_clean_stem(mvdr_name)

        mvdr_path = os.path.join(self.mvdr_dir, mvdr_name)
        clean_path = self.clean_index.get(clean_stem)

        if not os.path.isfile(mvdr_path):
            raise FileNotFoundError(f"Missing MVDR wav: {mvdr_path}")
        if clean_path is None or not os.path.isfile(clean_path):
            raise FileNotFoundError(
                f"Missing clean reference for '{mvdr_name}'. Expected a file named '{clean_stem}.wav' or '{clean_stem}.flac' "
                f"inside {self.clean_dir}"
            )

        mvdr_wav = _load_wav(mvdr_path, self.sr)
        clean_wav = _load_wav(clean_path, self.sr)

        # Ensure paired signals have identical length
        min_len = min(mvdr_wav.shape[-1], clean_wav.shape[-1])
        mvdr_wav = mvdr_wav[:min_len]
        clean_wav = clean_wav[:min_len]

        # Normalize both signals using the MVDR scale to preserve relative loudness.
        # This tends to behave better for perceptual metrics than normalizing each independently.
        scale = mvdr_wav.std() + 1e-8
        mvdr_wav = mvdr_wav / scale
        clean_wav = clean_wav / scale

        return mvdr_wav, clean_wav